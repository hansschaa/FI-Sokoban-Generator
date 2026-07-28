"""
experiment_ranking_quality.py
------------------------------
Experimento diagnostico para entender la baja calidad de ranking intra-bucket
del regresor en zonas de alta dificultad (Spearman ~0.13 en buckets 91+).

Protocolo experimental:
  - Train: folds 2-5 concatenados (~80% del dataset, sin solapamiento con eval)
  - Eval:  fold 1 (baseline ya conocido: MAE=5.656, Spearman(101+)=+0.217)
  - N_EPOCHS_EXP: 15 epocas cortas para comparacion rapida, no entrenamiento final

Variantes probadas:
  A) Baseline:      Huber + weight=1.0 para todos
  B) Upweight:      Huber + weight=3.0 para buckets >= 61
  C) Ranking Loss:  Huber + pairwise margin ranking loss (lambda=0.1)
  D) Both:          Upweight + Ranking Loss combinados

Uso:
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/experiment_ranking_quality.py
"""

import os, sys, time, json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.resnet import SokobanSEResNetRegressor
from data.board_utils import augment_tensor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuracion
N_EPOCHS_EXP    = 15
BATCH_SIZE      = 128
SPEARMAN_MIN_N  = 30
HARD_BUCKETS    = {"61_to_70", "71_to_80", "81_to_90", "91_to_100", "101_plus"}
HARD_WEIGHT     = 3.0
LAMBDA_RANK     = 0.1
MARGIN          = 0.05
N_PAIRS         = 256
EVAL_FOLD       = 1
TRAIN_FOLDS     = [2, 3, 4, 5]

with open(os.path.join(RESULTS_DIR, "best_hparams.json")) as f:
    cfg = json.load(f)["params"]
LR, WD, DROPOUT_P = cfg["lr"], cfg["weight_decay"], cfg["dropout_p"]

print("=" * 70)
print("  EXPERIMENTO: CALIDAD DE RANKING INTRA-BUCKET")
print("=" * 70)
print(f"  Dispositivo:  {device}")
print(f"  Train folds:  {TRAIN_FOLDS}  |  Eval fold: {EVAL_FOLD}")
print(f"  Epocas:       {N_EPOCHS_EXP}")
print(f"  LR={LR:.5f}  WD={WD:.6f}  drop={DROPOUT_P:.3f}  bs={BATCH_SIZE}")
print()

def recover_norm_stats(data_list):
    log_vals  = np.array([np.log1p(item["pushes_raw"])  for item in data_list])
    norm_vals = np.array([item["pushes_norm"] for item in data_list])
    A = np.stack([np.ones_like(norm_vals), norm_vals], axis=1)
    res = np.linalg.lstsq(A, log_vals, rcond=None)
    return float(res[0][0]), float(res[0][1])

print("  Cargando datos...")
train_raw_base = []
for fold in TRAIN_FOLDS:
    d = torch.load(os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt"),
                   map_location="cpu", weights_only=False)
    train_raw_base.extend(d)

eval_raw = torch.load(
    os.path.join(RESULTS_DIR, f"regressor_fold{EVAL_FOLD}_test.pt"),
    map_location="cpu", weights_only=False)

p_mean, p_std = recover_norm_stats(train_raw_base)
print(f"  Norm stats (de train): mean={p_mean:.3f}, std={p_std:.3f}")
print(f"  Train: {len(train_raw_base):,}  |  Eval: {len(eval_raw):,}")
print()

# Distribución de buckets difíciles en train
hard_n = sum(1 for item in train_raw_base if item.get("bucket","") in HARD_BUCKETS)
print(f"  Hard buckets en train (>=61): {hard_n:,} / {len(train_raw_base):,} = {hard_n/len(train_raw_base)*100:.1f}%")
very_hard_n = sum(1 for item in train_raw_base if item.get("bucket","") in {"91_to_100","101_plus"})
print(f"  Very hard (91+):              {very_hard_n:,} / {len(train_raw_base):,} = {very_hard_n/len(train_raw_base)*100:.1f}%")
print()


class WeightedRegressorDataset(Dataset):
    def __init__(self, data_list, p_mean, p_std, hard_weight=1.0):
        self.data      = [dict(item) for item in data_list]
        self.mean      = p_mean
        self.std       = p_std
        self.aug_factor = 8
        for item in self.data:
            item["pushes_norm"] = (np.log1p(item["pushes_raw"]) - p_mean) / (p_std + 1e-8)
            bk = item.get("bucket", "")
            item["_w"] = hard_weight if bk in HARD_BUCKETS else 1.0

    def __len__(self):
        return len(self.data) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx // self.aug_factor
        aug_idx  = idx %  self.aug_factor
        item     = self.data[orig_idx]
        aug_variants = list(augment_tensor(item["tensor"].numpy()))
        tensor_out   = torch.from_numpy(aug_variants[aug_idx].copy()).float()
        return (tensor_out,
                torch.tensor(item["pushes_norm"], dtype=torch.float32),
                torch.tensor(item["pushes_raw"],  dtype=torch.float32),
                torch.tensor(item["_w"],          dtype=torch.float32))

def collate_train(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]))


def pairwise_ranking_loss(pred, target, n_pairs=N_PAIRS, margin=MARGIN):
    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred.device)
    idx_i = torch.randint(0, n, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=pred.device)
    same  = (idx_i == idx_j)
    idx_j[same] = (idx_j[same] + 1) % n
    pi, pj = pred[idx_i], pred[idx_j]
    ti, tj = target[idx_i], target[idx_j]
    diff_target = ti - tj
    significant = diff_target.abs() > 0.01
    if not significant.any():
        return torch.tensor(0.0, device=pred.device)
    sign_target = torch.sign(diff_target[significant])
    pred_diff   = pi[significant] - pj[significant]
    return torch.relu(margin - pred_diff * sign_target).mean()


def evaluate_fold(model, eval_data, p_mean, p_std):
    model.eval()
    all_pred, all_gt, all_buckets = [], [], []
    with torch.no_grad():
        for item in eval_data:
            x = item["tensor"].float().unsqueeze(0).to(device)
            norm_pred = model(x).cpu().item()
            pred_real = np.expm1(norm_pred * p_std + p_mean)
            all_pred.append(pred_real)
            all_gt.append(item["pushes_raw"])
            all_buckets.append(item.get("bucket", "unknown"))

    all_pred    = np.array(all_pred)
    all_gt      = np.array(all_gt)
    all_buckets = np.array(all_buckets)

    mae  = float(np.abs(all_pred - all_gt).mean())
    rho, _ = spearmanr(all_gt, all_pred)

    bucket_rhos = {}
    bucket_ns   = {}
    for bk in sorted(set(all_buckets)):
        mask = (all_buckets == bk)
        n_bk = int(mask.sum())
        bucket_ns[bk] = n_bk
        if n_bk >= SPEARMAN_MIN_N and len(np.unique(all_gt[mask])) > 1:
            r, _ = spearmanr(all_gt[mask], all_pred[mask])
            bucket_rhos[bk] = float(r)
        else:
            bucket_rhos[bk] = float("nan")
    return mae, rho, bucket_rhos, bucket_ns


def run_experiment(variant_name, use_hard_weight, use_ranking_loss):
    print(f"\n{'─'*60}")
    print(f"  VARIANTE: {variant_name}")
    hard_w  = HARD_WEIGHT if use_hard_weight else 1.0
    dataset = WeightedRegressorDataset(train_raw_base, p_mean, p_std, hard_weight=hard_w)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, collate_fn=collate_train)

    model     = SokobanSEResNetRegressor(dropout_p=DROPOUT_P).to(device)
    criterion = nn.HuberLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_EXP)

    for epoch in range(1, N_EPOCHS_EXP + 1):
        t0 = time.time()
        model.train()
        total_h, total_r = 0.0, 0.0
        for x, p_norm, p_raw, weights in loader:
            x, p_norm, weights = x.to(device), p_norm.to(device), weights.to(device)
            optimizer.zero_grad()
            pred = model(x)
            huber = (criterion(pred, p_norm) * weights).mean()
            rank_t = pairwise_ranking_loss(pred.squeeze(), p_norm) if use_ranking_loss else torch.tensor(0.0)
            loss = huber + LAMBDA_RANK * rank_t
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_h += huber.item()
            total_r += rank_t.item() if use_ranking_loss else 0.0
        scheduler.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS_EXP:
            print(f"    Ep {epoch:02d}/{N_EPOCHS_EXP} | {time.time()-t0:.1f}s | "
                  f"Huber={total_h/len(loader):.4f} | Rank={total_r/len(loader):.4f}")

    mae, rho_global, bucket_rhos, bucket_ns = evaluate_fold(model, eval_raw, p_mean, p_std)
    print(f"  -> MAE={mae:.3f} | Spearman_global={rho_global:.4f}")
    print(f"  -> Spearman por bucket (n, ±CI95, valor):")
    for bk in sorted(bucket_rhos.keys()):
        r   = bucket_rhos[bk]
        n_b = bucket_ns.get(bk, 0)
        ci  = 1.96 / np.sqrt(max(n_b - 3, 1))
        r_str = f"{r:+.3f} ±{ci:.3f}" if not np.isnan(r) else "  n/a"
        marker = " <--" if bk in HARD_BUCKETS else ""
        print(f"       {bk:>12} (n={n_b:3d}): {r_str}{marker}")
    return {"mae": mae, "rho_global": rho_global, "bucket_rhos": bucket_rhos, "bucket_ns": bucket_ns}


if __name__ == "__main__":
    results = {}
    results["A_baseline"] = run_experiment("A: Baseline (Huber, weight=1)",
                                           use_hard_weight=False, use_ranking_loss=False)
    results["B_upweight"] = run_experiment("B: Upweight hard (w=3.0)",
                                           use_hard_weight=True, use_ranking_loss=False)
    results["C_ranking"]  = run_experiment("C: Pairwise ranking loss (λ=0.1)",
                                           use_hard_weight=False, use_ranking_loss=True)
    results["D_both"]     = run_experiment("D: Upweight + Ranking loss",
                                           use_hard_weight=True, use_ranking_loss=True)

    print("\n\n" + "=" * 70)
    print("  TABLA COMPARATIVA — control correcto: A_baseline (mismas epocas, mismo eval)")
    print(f"  Epocas por variante: {N_EPOCHS_EXP}")
    print("  NOTA: comparar contra 'CV completo' (MAE=5.656, Rho(101+)=+0.217) NO es valido")
    print("        porque el CV uso mas epocas y el modelo fue entrenado con mas datos.")
    print("="*70)

    # Mostrar n y CI para que las diferencias sean interpretables
    base_ns = results["A_baseline"]["bucket_ns"]
    ci_81  = 1.96 / np.sqrt(max(base_ns.get("81_to_90", 3) - 3, 1))
    ci_91  = 1.96 / np.sqrt(max(base_ns.get("91_to_100", 3) - 3, 1))
    ci_101 = 1.96 / np.sqrt(max(base_ns.get("101_plus", 3) - 3, 1))
    n_81   = base_ns.get("81_to_90", 0)
    n_91   = base_ns.get("91_to_100", 0)
    n_101  = base_ns.get("101_plus", 0)
    print(f"  Umbrales de significancia (95% CI half-width):")
    print(f"    81_to_90  (n={n_81:3d}): diferencia debe superar ±{ci_81:.3f} para ser señal")
    print(f"    91_to_100 (n={n_91:3d}): diferencia debe superar ±{ci_91:.3f}")
    print(f"    101_plus  (n={n_101:3d}): diferencia debe superar ±{ci_101:.3f}")
    print()

    base  = results["A_baseline"]["bucket_rhos"]
    print(f"  {'Variante':>25}  {'MAE':>6}  {'Rho_global':>11}  "
          f"{'Rho_81-90 (delta)':>20}  {'Rho_101+ (delta)':>19}")
    print(f"  {'-'*85}")
    for name, res in results.items():
        bk      = res["bucket_rhos"]
        r81_v   = bk.get("81_to_90", float("nan"))
        r101_v  = bk.get("101_plus", float("nan"))
        r81_b   = base.get("81_to_90", float("nan"))
        r101_b  = base.get("101_plus", float("nan"))

        if not np.isnan(r81_v) and not np.isnan(r81_b):
            d81 = r81_v - r81_b
            sig81 = "*" if abs(d81) > ci_81 else " "
            r81_str = f"{r81_v:+.3f} (delta={d81:+.3f}){sig81}"
        else:
            r81_str = "  n/a"

        if not np.isnan(r101_v) and not np.isnan(r101_b):
            d101 = r101_v - r101_b
            sig101 = "*" if abs(d101) > ci_101 else " "
            r101_str = f"{r101_v:+.3f} (delta={d101:+.3f}){sig101}"
        else:
            r101_str = "  n/a"

        print(f"  {name:>25}  {res['mae']:6.3f}  {res['rho_global']:+11.4f}  "
              f"{r81_str:>20}  {r101_str:>19}")

    print()
    print("  * = diferencia supera CI95 (posiblemente señal real, no ruido de muestra)")
    print()
    print("  INTERPRETACION CORRECTA:")
    print("  - Si ninguna diferencia marcada con * -> experimento insuficiente para")
    print("    distinguir las variantes. Conclusion: limite de representacion probable,")
    print("    no volumen ni loss. Minar mas datos NO esta justificado aun.")
    print("  - Si B(*) mejora 81-90 pero destruye 101+ -> trade-off, no mejora limpia.")
    print("    Investigar upweighting mas graduado o focal loss antes de minar mas.")
    print("  - Si C(*) mejora zonas hard -> cambiar loss en modelo de produccion.")
