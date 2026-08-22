"""
train_final_path_consistency_v2.py
────────────────────────────────────
Entrenamiento final del regresor de Path Consistency sobre el 100% de los
datos disponibles (todos los folds combinados, sin reservar test).

Diferencias respecto a train_final_path_consistency.py:
  - Acepta hiperparámetros como argumentos CLI (no hardcodeados).
  - Si no se pasan hparams, lee de best_hparams_path_consistency_v2.json.
  - Logging explícito de pipeline hash + hparams al arranque.
  - Guarda el modelo con SHA256 checksum al finalizar.

Uso:
  # Con hparams del Optuna v2 (recomendado)
  python surrogate_models/train_final_path_consistency_v2.py

  # Sobreescribiendo hparams manualmente
  python surrogate_models/train_final_path_consistency_v2.py \\
      --lr 0.001 --weight-decay 1e-5 --dropout 0.1 \\
      --batch-size 256 --alpha 0.08 --margin 0.05 \\
      --epochs 60
"""

import os, sys, json, copy, time, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import spearmanr
import numpy as np

from models.resnet import SokobanSEResNetRegressor
from train_final_path_consistency import PathConsistencyDataset, RegressorDataset, RESULTS_DIR

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Pipeline hash ──────────────────────────────────────────────────────────────
def compute_pipeline_hash():
    h = hashlib.sha256()
    for p in [os.path.join(SCRIPT_DIR, "models", "resnet.py"),
              os.path.join(SCRIPT_DIR, "train_final_path_consistency.py"),
              os.path.join(SCRIPT_DIR, "prepare_path_consistency.py")]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(p.encode()); h.update(f.read())
    return h.hexdigest()[:16]

PIPELINE_HASH = compute_pipeline_hash()

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""): h.update(block)
    return h.hexdigest()
# ──────────────────────────────────────────────────────────────────────────────


def load_hparams(args) -> dict:
    """Carga hiperparámetros: CLI tiene prioridad > JSON v2 > JSON original."""
    # Valores por defecto desde JSON
    defaults = {
        "lr": None, "weight_decay": None, "dropout_p": None,
        "batch_size": None, "alpha": None, "margin": None,
    }

    # Intentar cargar desde JSON v2 primero, luego original
    for fname in ["best_hparams_path_consistency_v2.json",
                  "best_hparams_path_consistency.json"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            params = data.get("params", data.get("best_params", {}))
            defaults.update({k: v for k, v in params.items() if k in defaults})
            print(f"  📂 Hiperparámetros base cargados desde: {fname}")
            break

    # CLI sobreescribe
    if args.lr           is not None: defaults["lr"]           = args.lr
    if args.weight_decay is not None: defaults["weight_decay"] = args.weight_decay
    if args.dropout      is not None: defaults["dropout_p"]    = args.dropout
    if args.batch_size   is not None: defaults["batch_size"]   = args.batch_size
    if args.alpha        is not None: defaults["alpha"]        = args.alpha
    if args.margin       is not None: defaults["margin"]       = args.margin

    # Verificar que todos los hparams están definidos
    missing = [k for k, v in defaults.items() if v is None]
    if missing:
        raise ValueError(f"Hiperparámetros sin definir: {missing}. "
                         f"Proporcionalos via CLI o asegurá que exista el JSON.")
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento final del regresor de Path Consistency (100% datos)")
    # Hparams opcionales (sobreescriben el JSON)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout",      type=float, default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--alpha",        type=float, default=None)
    parser.add_argument("--margin",       type=float, default=None)
    parser.add_argument("--epochs",       type=int,   default=60)
    parser.add_argument("--max_route_distance", type=int, default=1)
    parser.add_argument("--test-fold",    type=int,   default=1,
                        help="Fold usado para test (los demás 4 se usan para train)")
    parser.add_argument("--output",       type=str,
                        default=None,
                        help="Ruta de salida (por defecto auto-generada)")
    parser.add_argument("--restart", action="store_true",
                        help="Ignora checkpoints y empieza desde cero")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    hparams = load_hparams(args)
    lr           = float(hparams["lr"])
    weight_decay = float(hparams["weight_decay"])
    dropout_p    = float(hparams["dropout_p"])
    batch_size   = int(hparams["batch_size"])
    alpha        = float(hparams["alpha"])
    margin       = float(hparams["margin"])
    # batch efectivo para pares (cada par usa 2 items)
    loader_batch = max(1, batch_size // 2)

    print("=" * 70)
    print("  ENTRENAMIENTO FINAL: PATH CONSISTENCY REGRESSOR (100% datos)")
    print("=" * 70)
    print(f"  Pipeline Hash : {PIPELINE_HASH}")
    print(f"  Device        : {device.type.upper()} "
          f"({'CUDA — ' + torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Épocas        : {args.epochs}")
    print(f"  Hiperparámetros:")
    for k, v in hparams.items():
        print(f"    {k:15s}: {v}")
    print(f"  Output        : {args.output}")
    print("=" * 70)

    folds_to_use = [f for f in range(1, 6) if f != args.test_fold]

    # ── Combinar todos los folds de path_consistency ─────────────────────────
    print("\n  Cargando datasets de Path Consistency...")
    all_pc_datasets = []
    for fold in folds_to_use:
        pc_path = os.path.join(RESULTS_DIR, "path_consistency",
                               f"path_fold{fold}_train.pt")
        if not os.path.exists(pc_path):
            print(f"  ⚠️  No existe: {pc_path}. Saltando fold {fold}.")
            continue
        ds = PathConsistencyDataset(fold, augment=True, max_route_distance=args.max_route_distance)
        all_pc_datasets.append(ds)
        print(f"    Fold {fold}: {len(ds):,} pares cargados")

    if not all_pc_datasets:
        raise FileNotFoundError("No se encontró ningún dataset de path consistency.")

    fold_loaders = []
    total_pairs = 0
    for ds in all_pc_datasets:
        loader = DataLoader(ds, batch_size=loader_batch, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
        fold_loaders.append(loader)
        total_pairs += len(ds)
    print(f"  Total pares combinados (Train): {total_pairs:,}")

    # ── Test DataLoader ───────────────────────────────────────────────────────
    print(f"\n  Cargando Test Fold {args.test_fold}...")
    test_pc_path = os.path.join(RESULTS_DIR, "path_consistency", f"path_fold{args.test_fold}_train.pt")
    if not os.path.exists(test_pc_path):
        test_pc_path_v2 = os.path.join(RESULTS_DIR, "path_consistency_v2", f"path_fold{args.test_fold}_train.pt")
        if os.path.exists(test_pc_path_v2):
            test_pc_path = test_pc_path_v2
        else:
            raise FileNotFoundError(f"No se encontró el test fold: {test_pc_path}")
    test_ds = PathConsistencyDataset(args.test_fold, augment=False, max_route_distance=args.max_route_distance)
    test_loader = DataLoader(test_ds, batch_size=loader_batch, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=False)
    print(f"    Test Fold {args.test_fold}: {len(test_ds):,} pares cargados")

    # Stats de normalización (usados consistentemente desde el fold 1 V1)
    p_mean = 3.4614
    p_std  = 0.8732
    print(f"\n  Stats log1p: mean={p_mean:.4f}, std={p_std:.4f}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model      = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
    huber_fn   = nn.HuberLoss(delta=1.0)
    ranking_fn = nn.MarginRankingLoss(margin=margin)
    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    ckpt_path   = args.output.replace(".pt", "_ckpt.pt")
    start_epoch = 1
    best_loss   = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    PATIENCE = 20

    if args.restart and os.path.exists(ckpt_path):
        print(f"\n  ⚠️  --restart: borrando checkpoint {ckpt_path}")
        os.remove(ckpt_path)

    if os.path.exists(ckpt_path):
        print(f"\n  🔄 Reanudando desde: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch  = ckpt['epoch'] + 1
        best_loss    = ckpt['best_loss']
        best_weights = ckpt['best_weights']
        patience_ctr = ckpt['patience_ctr']
        print(f"    Reanudando desde época {start_epoch} | best_loss={best_loss:.4f}")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    print(f"\n  🚀 Iniciando entrenamiento ({start_epoch}..{args.epochs})...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_huber, total_margin = 0.0, 0.0, 0.0

        from tqdm import tqdm
        
        # Calculamos el total de batches sumando la longitud de cada loader
        total_batches_epoch = sum(len(loader) for loader in fold_loaders)
        
        # Envolvemos los loaders en tqdm para ver el progreso real
        pbar = tqdm(total=total_batches_epoch, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)

        for loader in fold_loaders:
            for batch in loader:
                x1     = batch['tensor1'].to(device)
                x2     = batch['tensor2'].to(device)
                p1_raw = batch['pushes1'].float().to(device)
                p2_raw = batch['pushes2'].float().to(device)
                weight = batch['weight'].to(device)

                # Normalización del target
                y_target = (torch.log1p(p1_raw) - p_mean) / p_std
                
                optimizer.zero_grad()
                pred1 = model(x1).squeeze(-1)
                pred2 = model(x2).squeeze(-1)

                loss_huber = (huber_fn(pred1, y_target) * weight).mean()
                
                # Loss margin: max(0, margin - sign(y1 - y2) * (pred1 - pred2))
                diff_pred = pred1 - pred2
                diff_true = p1_raw - p2_raw
                loss_margin = torch.clamp(margin - (diff_pred * torch.sign(diff_true)), min=0.0)
                loss_margin = (loss_margin * weight).mean()
                
                loss = loss_huber + alpha * loss_margin
                loss.backward()
                optimizer.step()

                total_loss   += loss.item()
                total_huber  += loss_huber.item()
                total_margin += loss_margin.item()
                pbar.update(1)
        
        pbar.close()

        # Calculamos los promedios dividiendo por el total real de batches
        total_batches_epoch = sum(len(loader) for loader in fold_loaders)
        avg_loss   = total_loss / total_batches_epoch
        avg_huber  = total_huber / total_batches_epoch
        avg_margin = total_margin / total_batches_epoch
        scheduler.step()

        # ── Validación (TEST MAE) ───────────────────────────────────────────────
        model.eval()
        test_loss, test_mae, test_huber, test_margin = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in test_loader:
                x1     = batch['tensor1'].to(device)
                x2     = batch['tensor2'].to(device)
                p1_raw = batch['pushes1'].float().to(device)
                p2_raw = batch['pushes2'].float().to(device)
                weight = batch['weight'].to(device)

                y_target = (torch.log1p(p1_raw) - p_mean) / p_std
                
                pred1 = model(x1).squeeze(-1)
                pred2 = model(x2).squeeze(-1)

                loss_huber = (huber_fn(pred1, y_target) * weight).mean()
                diff_pred = pred1 - pred2
                diff_true = p1_raw - p2_raw
                loss_margin = torch.clamp(margin - (diff_pred * torch.sign(diff_true)), min=0.0)
                loss_margin = (loss_margin * weight).mean()
                
                loss = loss_huber + alpha * loss_margin
                
                # Desnormalizar predicción para calcular MAE real en pushes
                pred1_pushes = torch.expm1(pred1 * p_std + p_mean)
                mae = torch.abs(pred1_pushes - p1_raw).mean()

                test_loss   += loss.item()
                test_huber  += loss_huber.item()
                test_margin += loss_margin.item()
                test_mae    += mae.item()

        avg_test_loss   = test_loss / len(test_loader)
        avg_test_huber  = test_huber / len(test_loader)
        avg_test_margin = test_margin / len(test_loader)
        avg_test_mae    = test_mae / len(test_loader)

        scheduler.step()

        tag = ""
        # Guardamos el mejor modelo en base al MAE TEST
        if avg_test_mae < best_loss:
            best_loss    = avg_test_mae
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = " ★"
        else:
            patience_ctr += 1

        elapsed = time.time() - t0
        print(f"  Ep {epoch:03d}/{args.epochs} | {elapsed:.1f}s "
              f"| Train L: {avg_loss:.4f} | Test MAE: {avg_test_mae:.4f} (H: {avg_test_huber:.4f} M: {avg_test_margin:.4f})"
              f" | patience: {patience_ctr}/{PATIENCE}{tag}")

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_weights': best_weights,
            'patience_ctr': patience_ctr,
            'hparams': hparams,
            'pipeline_hash': PIPELINE_HASH,
        }, ckpt_path)

        if patience_ctr >= PATIENCE:
            print(f"\n  🛑 Early stopping en época {epoch}.")
            break

    # ── Guardar modelo final ──────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    torch.save(best_weights, args.output)
    sha256 = compute_sha256(args.output)

    # Limpiar checkpoint
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    # Guardar metadata
    meta = {
        "pipeline_hash": PIPELINE_HASH,
        "sha256": sha256,
        "hparams": hparams,
        "epochs_trained": epoch,
        "best_train_loss": best_loss,
    }
    meta_path = args.output.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  ✅ Modelo final guardado en: {args.output}")
    print(f"  SHA256 : {sha256}")
    print(f"  Meta   : {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
