"""
train_ranking_loss.py
---------------------
Experimento A: Entrena el regresor SE-ResNet con una loss compuesta
    Huber + MarginRankingLoss (pares intra-bucket).

La arquitectura es IDÉNTICA a train_final_surrogates.py — la única diferencia
es la función de pérdida. Esto permite aislar la contribución del Ranking Loss
sin confundir variables (condición del paper).

Uso:
    python3 train_ranking_loss.py --folds 1,2,3,4,5 --alpha 0.5 --margin 0.1
    python3 train_ranking_loss.py --folds 1 --alpha 0.3 --margin 0.05 --epochs 10  # Grid search rápido

Guarda pesos en: results/ranking_regressor_fold{k}.pt  (separados del baseline)
"""

import sys, os, json, copy, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import spearmanr

from models.resnet import SokobanSEResNetRegressor, SokobanSEResNetRegressorSpatial

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (idéntico al baseline)
# ─────────────────────────────────────────────────────────────────────────────
class RegressorDatasetRanking(Dataset):
    """
    Mismo layout que RegressorDataset, con el campo 'bucket' expuesto
    para que el sampler de pares pueda usarlo sin mirar el tensor.
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['tensor'].float(),
            torch.tensor(item['pushes_norm'], dtype=torch.float32),
            torch.tensor(item['pushes_raw'],  dtype=torch.float32),
            torch.tensor(item.get('weight', 1.0), dtype=torch.float32),
            item['bucket'],   # str, e.g. "91_to_100"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE PARES INTRA-BUCKET
# ─────────────────────────────────────────────────────────────────────────────
def build_ranking_pairs(p_pred, p_norm, buckets, min_diff=2.0 / 50.0, max_pairs=256):
    """
    Construye pares (i, j) intra-bucket con |p_norm_i - p_norm_j| >= min_diff.

    Devuelve:
      pred_i, pred_j : tensores de predicciones para cada par
      target         : +1 si p_norm[i] > p_norm[j], -1 en caso contrario
      pair_counts    : dict {bucket: n_pares}  ← para diagnóstico de sparsity
    """
    pairs_i, pairs_j, targets = [], [], []
    pair_counts = {}  # bucket → nº de pares formados

    bucket_to_indices = {}
    for idx, b in enumerate(buckets):
        bucket_to_indices.setdefault(b, []).append(idx)

    for b, indices in bucket_to_indices.items():
        bucket_pairs = 0
        if len(indices) < 2:
            pair_counts[b] = 0
            continue
        indices = list(indices)
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                i, j = indices[ii], indices[jj]
                diff = (p_norm[i] - p_norm[j]).item()
                if abs(diff) < min_diff:
                    continue
                pairs_i.append(i)
                pairs_j.append(j)
                targets.append(1.0 if diff > 0 else -1.0)
                bucket_pairs += 1
        pair_counts[b] = bucket_pairs

    if not pairs_i:
        return None, None, None, pair_counts

    # Submuestrear si hay demasiados pares (costo cuadrático)
    if len(pairs_i) > max_pairs:
        chosen = np.random.choice(len(pairs_i), max_pairs, replace=False)
        pairs_i = [pairs_i[k] for k in chosen]
        pairs_j = [pairs_j[k] for k in chosen]
        targets  = [targets[k]  for k in chosen]

    pi = torch.stack([p_pred[k] for k in pairs_i])
    pj = torch.stack([p_pred[k] for k in pairs_j])
    tg = torch.tensor(targets, dtype=torch.float32, device=p_pred.device)
    return pi, pj, tg, pair_counts


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN INTRA-BUCKET (extra vs baseline)
# ─────────────────────────────────────────────────────────────────────────────
def eval_spearman_by_bucket(p_raw_list, p_pred_list, bucket_list):
    """
    Calcula Spearman GLOBAL y por bucket. Devuelve dict {bucket: rho} + "global".
    """
    from collections import defaultdict
    groups = defaultdict(lambda: ([], []))
    for raw, pred, b in zip(p_raw_list, p_pred_list, bucket_list):
        groups[b][0].append(raw)
        groups[b][1].append(pred)

    result = {}
    global_rho, _ = spearmanr(p_raw_list, p_pred_list)
    result["global"] = global_rho

    def bucket_sort_key(b):
        if b == '101_plus': return 101
        return int(b.split('_')[0])

    for b in sorted(groups.keys(), key=bucket_sort_key):
        raws, preds = groups[b]
        if len(raws) < 2:
            result[b] = float('nan')
        else:
            rho, _ = spearmanr(raws, preds)
            result[b] = rho
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def train_ranking_regressor(folds_to_run, alpha, margin, min_diff_norm, max_pairs,
                             max_epochs, restart, arch="baseline"):
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    if not os.path.exists(hparams_path):
        print("❌ Error: No se encontró best_hparams.json")
        return

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr           = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p    = cfg["dropout_p"]
    batch_size   = int(cfg["batch_size"])

    huber_criterion  = nn.HuberLoss(reduction='none')
    ranking_criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')

    print("\n" + "="*70)
    print(f"  Experimento   : Ranking Loss con Arquitectura = {arch.upper()}")
    print(f"  Dispositivo   : {device.type.upper()}")
    print(f"  α             : {alpha}  (1-α)*Huber + α*MarginRanking")
    print(f"  margin        : {margin}")
    print(f"  min_diff_norm : {min_diff_norm}")
    print(f"  max_pairs/bat : {max_pairs}")
    print(f"  lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, bs={batch_size}\n")

    test_sp_91 = float('nan')
    fold_sp91_values = []  # para criterio multi-fold
    fold_maes = []

    for fold in folds_to_run:
        train_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_train.pt")
        val_path   = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_val.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")

        if not os.path.exists(train_path):
            print(f"⚠️  Saltando Fold {fold}: No existe {train_path}")
            continue

        print(f"\n[{'─'*50}]")
        print(f"  INICIANDO FOLD {fold}/5 (Ranking Loss)")
        print(f"[{'─'*50}]")

        train_data = torch.load(train_path, weights_only=False)
        val_data   = torch.load(val_path,   weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)
        stats      = torch.load(stats_path, weights_only=False)
        p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]

        train_loader = DataLoader(RegressorDatasetRanking(train_data),
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(RegressorDatasetRanking(val_data),
                                  batch_size=256, shuffle=False,
                                  num_workers=0, pin_memory=True)
        test_loader  = DataLoader(RegressorDatasetRanking(test_data),
                                  batch_size=256, shuffle=False,
                                  num_workers=0, pin_memory=True)

        # Seleccionar Arquitectura
        if arch == "spatial":
            model = SokobanSEResNetRegressorSpatial(dropout_p=dropout_p).to(device)
        else:
            model = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
            
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_mae = float("inf")
        best_weights_mae = copy.deepcopy(model.state_dict())
        best_sp91 = -float("inf")
        best_weights_sp91 = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_ranking_{arch}_regressor_fold{fold}.pt")
        start_epoch = 1

        if restart and os.path.exists(ckpt_path):
            print(f"  -> ⚠️  Bandera --restart: borrando checkpoint anterior.")
            os.remove(ckpt_path)

        if os.path.exists(ckpt_path):
            print(f"  -> 🔄 Reanudando desde checkpoint: {os.path.basename(ckpt_path)}")
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mae = ckpt['best_mae']
            best_weights_mae = ckpt['best_weights']
            best_sp91 = ckpt.get('best_sp91', -float('inf'))
            best_weights_sp91 = ckpt.get('best_weights_sp91', best_weights_mae)
            patience_ctr = ckpt['patience_ctr']

        for epoch in range(start_epoch, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss_total = 0.0
            train_loss_huber = 0.0
            train_loss_rank  = 0.0
            n_batches = 0
            epoch_pair_counts = {}  # acumula pares por bucket a lo largo de la época

            for tensors, p_norm, p_raw, weights, buckets in train_loader:
                tensors = tensors.to(device)
                p_norm  = p_norm.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()
                p_pred = model(tensors)

                # ── Huber loss (ponderado por sample weight) ──────────────
                loss_huber = (huber_criterion(p_pred, p_norm) * weights).mean()

                # ── Ranking loss intra-bucket ─────────────────────────────
                # Cortocircuito: si alpha=0 (modo Spatial puro / Huber solo),
                # no construir pares — es puro gasto de CPU sin efecto en el gradiente.
                if alpha > 0:
                    pi, pj, tg, pair_counts = build_ranking_pairs(
                        p_pred, p_norm, buckets,
                        min_diff=min_diff_norm, max_pairs=max_pairs
                    )
                    if pi is not None:
                        loss_rank = ranking_criterion(pi, pj, tg.to(device))
                    else:
                        loss_rank = torch.tensor(0.0, device=device)
                        pair_counts = {}
                    # Acumular conteo de pares por bucket para diagnóstico de sparsity
                    for b, cnt in pair_counts.items():
                        epoch_pair_counts[b] = epoch_pair_counts.get(b, 0) + cnt
                else:
                    loss_rank = torch.tensor(0.0, device=device)

                loss = (1 - alpha) * loss_huber + alpha * loss_rank
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss_total += loss.item()
                train_loss_huber += loss_huber.item()
                train_loss_rank  += loss_rank.item()
                n_batches += 1

            train_loss_total /= n_batches
            train_loss_huber /= n_batches
            train_loss_rank  /= n_batches

            # ── Evaluación en Validación ───────────────────────────────────
            model.eval()
            total_mae_val, n_val = 0.0, 0
            all_p_pred_val, all_p_raw_val, all_buckets_val = [], [], []
            with torch.no_grad():
                for tensors, _, p_raw, _, buckets in val_loader:
                    tensors = tensors.to(device)
                    p_pred_val = model(tensors)
                    p_desnorm = p_pred_val.cpu() * p_std + p_mean
                    p_desnorm_real = torch.expm1(p_desnorm)
                    total_mae_val += torch.abs(p_desnorm_real - p_raw).sum().item()
                    n_val += len(p_raw)
                    all_p_pred_val.extend(p_desnorm_real.view(-1).tolist())
                    all_p_raw_val.extend(p_raw.view(-1).tolist())
                    all_buckets_val.extend(list(buckets))

            val_mae = total_mae_val / n_val
            spearman_by_bucket = eval_spearman_by_bucket(
                all_p_raw_val, all_p_pred_val, all_buckets_val
            )
            val_spearman_global = spearman_by_bucket.get("global", float('nan'))
            val_spearman_91     = np.nanmean([
                spearman_by_bucket.get(b, float('nan'))
                for b in ('91_to_100', '101_plus')
            ])

            scheduler.step()
            elapsed = time.time() - t0

            # ── Log de pares por bucket (diagnóstico de sparsity) ─────────
            def bkey(b):
                if b == '101_plus': return 101
                try: return int(b.split('_')[0])
                except: return 0
            pair_log = "  ".join(
                f"{b}:{epoch_pair_counts.get(b,0)}"
                for b in sorted(epoch_pair_counts, key=bkey)
            )

            tag = ""
            if val_mae < best_mae:
                best_mae = val_mae
                best_weights_mae = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                tag += " ★MAE"
            else:
                patience_ctr += 1

            if not np.isnan(val_spearman_91) and val_spearman_91 > best_sp91:
                best_sp91 = val_spearman_91
                best_weights_sp91 = copy.deepcopy(model.state_dict())
                tag += " ★Sp91"

            print(
                f"  Ep {epoch:02d} | {elapsed:.1f}s "
                f"| Loss {train_loss_total:.4f} (H={train_loss_huber:.4f} R={train_loss_rank:.4f}) "
                f"| MAE {val_mae:.2f} | Sp_global {val_spearman_global:.3f} "
                f"| Sp_91+ {val_spearman_91:.3f}{tag}\n"
                f"         Pares/bucket: [{pair_log}]"
            )

            if patience_ctr >= 15:
                print(f"  🛑 Early Stopping en época {epoch}.")
                break

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'best_weights': best_weights_mae,
                'best_sp91': best_sp91,
                'best_weights_sp91': best_weights_sp91,
                'patience_ctr': patience_ctr,
            }, ckpt_path)

        # ── Test Ciego ─────────────────────────────────────────────────────
        if best_sp91 > -float('inf'):
            model.load_state_dict(best_weights_sp91)
            final_weights = best_weights_sp91
            print(f"\n  [INFO] Test evaluado con el mejor checkpoint por Sp_91+ (Val Sp_91+ = {best_sp91:.3f})")
        else:
            model.load_state_dict(best_weights_mae)
            final_weights = best_weights_mae
            print(f"\n  [INFO] Test evaluado con el mejor checkpoint por MAE (Val MAE = {best_mae:.2f})")
            
        model.eval()
        total_mae_test, n_test = 0.0, 0
        all_p_pred_test, all_p_raw_test, all_buckets_test = [], [], []
        with torch.no_grad():
            for tensors, _, p_raw, _, buckets in test_loader:
                tensors = tensors.to(device)
                p_pred_t = model(tensors)
                p_desnorm = p_pred_t.cpu() * p_std + p_mean
                p_desnorm_real = torch.expm1(p_desnorm)
                total_mae_test += torch.abs(p_desnorm_real - p_raw).sum().item()
                n_test += len(p_raw)
                all_p_pred_test.extend(p_desnorm_real.view(-1).tolist())
                all_p_raw_test.extend(p_raw.view(-1).tolist())
                all_buckets_test.extend(list(buckets))

        test_mae = total_mae_test / n_test
        test_spearman_by_bucket = eval_spearman_by_bucket(
            all_p_raw_test, all_p_pred_test, all_buckets_test
        )
        test_sp_global = test_spearman_by_bucket.get("global", float('nan'))
        test_sp_91     = np.nanmean([
            test_spearman_by_bucket.get(b, float('nan'))
            for b in ('91_to_100', '101_plus')
        ])

        fold_maes.append(test_mae)
        fold_sp91_values.append(test_sp_91)
        save_path = os.path.join(RESULTS_DIR, f"ranking_{arch}_regressor_fold{fold}.pt")
        torch.save(final_weights, save_path)

        print(f"\n  ✅ Fold {fold} → {os.path.basename(save_path)}")
        print(f"     MAE Val: {best_mae:.2f}  |  MAE TEST: {test_mae:.2f}")
        print(f"     Spearman GLOBAL (test): {test_sp_global:.3f}")
        print(f"     Spearman 91+    (test): {test_sp_91:.3f}")
        print(f"     Desglose por bucket:")
        def bucket_sort_key(b):
            if b == '101_plus': return 101
            return int(b.split('_')[0])
        for b in sorted(test_spearman_by_bucket.keys(), key=lambda b: bucket_sort_key(b) if b != 'global' else -1):
            if b == 'global': continue
            print(f"       {b:<15}: {test_spearman_by_bucket[b]:.3f}")

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    if len(fold_maes) > 1:
        print(f"\n  🏆 RANKING REGRESSOR (5-FOLD CV): MAE = {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} empujes")

    # ── Criterio de decisión (consciente de arch y alpha) ─────────────────────
    print("\n" + "="*70)
    print(f"  CRITERIO DE DECISIÓN  [arch={arch.upper()}  α={alpha}]")
    print("="*70)
    valid_sp91 = [v for v in fold_sp91_values if not np.isnan(v)]
    if not valid_sp91:
        print("  ⚠️  No hay folds válidos para evaluar.")
    else:
        sp91_mean = float(np.mean(valid_sp91))
        sp91_std  = float(np.std(valid_sp91)) if len(valid_sp91) > 1 else float('nan')
        std_str   = f" ± {sp91_std:.3f}" if not np.isnan(sp91_std) else ""
        print(f"  Sp_91+: {sp91_mean:.3f}{std_str}  (n={len(valid_sp91)} fold{'s' if len(valid_sp91)>1 else ''})")

        GBM_FLOOR = 0.38

        if sp91_mean >= GBM_FLOOR:
            print(f"  ✅ Sp_91+ ≥ {GBM_FLOOR} (GBM floor). Lanzar 5 folds completos → candidato paper.")
        elif sp91_mean >= 0.30:
            if arch == "spatial":
                print(f"  ✅ Sp_91+ ≥ 0.30 con Spatial Neck. Considerar lanzar 5 folds completos.")
                if alpha == 0:
                    print("     → Probar Spatial + Ranking α=0.10-0.15 como ajuste fino.")
            else:
                print(f"  ⚠️  Sp_91+ ≥ 0.30 con baseline. Buen resultado — combinar con Spatial Neck.")
        elif sp91_mean >= 0.20:
            if arch == "spatial" and alpha == 0:
                print(f"  ⚠️  Sp_91+ ∈ [0.20, 0.30) con Spatial puro.")
                print("     → Si la curva no se agotó (no disparó early stopping): extender épocas.")
                print("     → Si se agotó: probar Spatial + Ranking α=0.10-0.15 como ajuste fino.")
            elif arch == "spatial" and alpha > 0:
                print(f"  ⚠️  Sp_91+ ∈ [0.20, 0.30) con Spatial+Ranking α={alpha}.")
                print(f"     → Spatial puro (α=0) dio ~0.215. α={alpha} no mejoró sobre α=0.")
                print("     → Bajar α (0.05-0.10) o descartar Ranking sobre Spatial.")
            else:
                print(f"  ⚠️  Sp_91+ ∈ [0.20, 0.30) con baseline. Cambiar a --arch spatial.")
        elif sp91_mean >= 0.13:
            if arch == "spatial":
                print(f"  ⚠️  Sp_91+ ∈ [0.13, 0.20) con Spatial — por encima del baseline (~0.13).")
                if alpha > 0:
                    print(f"     → Spatial puro (α=0) superó esta combinación. Reducir α o usar α=0.")
            else:
                print(f"  ⚠️  Sp_91+ ∈ [0.13, 0.20). Mejora marginal sobre baseline.")
                print("     → Cambiar a --arch spatial para atacar el cuello de botella estructural.")
        else:
            if arch == "baseline" and alpha > 0:
                print(f"  ❌ Sp_91+ < 0.13 con Ranking Loss (baseline). Ranking Loss no penetra.")
                print("     → Causa: sparsity de pares en buckets difíciles O cuello 1×1.")
                print("     → Verificar log de Pares/bucket por época.")
                print("     → Cambiar a --arch spatial --alpha 0 (Spatial puro).")
            else:
                print(f"  ❌ Sp_91+ < 0.13. Sin mejora sobre baseline. Revisar configuración.")



# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Experimento A: Ranking Loss para SE-ResNet Regresor"
    )
    parser.add_argument("--folds", type=str, default="1,2,3,4,5",
                        help="Folds separados por coma (ej. 1 o 1,2,3)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Peso del ranking loss (0-1). 0 = solo Huber, 1 = solo Ranking.")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Margen de MarginRankingLoss en espacio normalizado.")
    parser.add_argument("--min-diff", type=float, default=0.02,
                        help="Diferencia mínima |p_norm_i - p_norm_j| para formar un par.")
    parser.add_argument("--max-pairs", type=int, default=256,
                        help="Máximo de pares por mini-batch.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Máximo de épocas de entrenamiento.")
    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "spatial"],
                        help="Arquitectura a usar: 'baseline' (AdaptiveAvgPool 1x1) o 'spatial' (AdaptiveAvgPool 3x3).")
    parser.add_argument("--restart", action="store_true",
                        help="Borra checkpoints y empieza desde cero.")

    args = parser.parse_args()
    folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    train_ranking_regressor(
        folds_to_run=folds_to_run,
        alpha=args.alpha,
        margin=args.margin,
        min_diff_norm=args.min_diff,
        max_pairs=args.max_pairs,
        max_epochs=args.epochs,
        restart=args.restart,
        arch=args.arch
    )


if __name__ == "__main__":
    main()
