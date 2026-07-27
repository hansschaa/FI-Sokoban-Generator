"""
evaluate_regressor_cv.py
--------------------------
Evaluacion completa 5-fold CV del regresor final SE-ResNet.
Reporta: MAE, RMSE, Spearman, desglose por bucket de dificultad
         con media +- std entre folds.

Definiciones:
  - MAE:      Mean Absolute Error en pushes reales (escala original)
  - RMSE:     Root Mean Squared Error en pushes reales
  - Spearman: Correlacion de ranking entre predicciones y ground truth
  - Bucket:   Agrupacion por dificultad (pushes reales) del puzzle

Normalizacion:
  - pushes_norm_i = (log1p(pushes_raw_i) - mean) / std
  - Se recuperan mean y std por fold via regresion lineal sobre los pares
    (log1p(pushes_raw), pushes_norm) del test set.

Uso:
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/evaluate_regressor_cv.py
"""

import os
import torch
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from models.resnet import SokobanSEResNetRegressor

OPTUNA_BEST_MAE = 5.43   # Mejor trial Optuna v4 (val Fold 1, sesgo optimista)
N_FOLDS    = 5
BATCH_SIZE = 256
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  EVALUACION CROSS-VALIDATION: REGRESOR (SE-ResNet)")
print("=" * 70)
print(f"  Dispositivo: {device}")
print(f"  Ref. Optuna (val, sesgo optimista): MAE = {OPTUNA_BEST_MAE:.2f} pushes")
print()
print("  Metrica principal: MAE en pushes reales (escala original)")
print()

class RegressorTestDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (item["tensor"].float(),
                torch.tensor(item["pushes_norm"], dtype=torch.float32),
                torch.tensor(item["pushes_raw"],  dtype=torch.float32),
                str(item.get("bucket", "unknown")))

def collate_fn(batch):
    tensors   = torch.stack([b[0] for b in batch])
    p_norm    = torch.stack([b[1] for b in batch])
    p_raw     = torch.stack([b[2] for b in batch])
    buckets   = [b[3] for b in batch]
    return tensors, p_norm, p_raw, buckets

def recover_norm_stats(data_list):
    """Recupera mean y std de normalizacion via regresion lineal
       sobre pares (log1p(pushes_raw), pushes_norm) del test set."""
    log_vals  = np.array([np.log1p(item["pushes_raw"])  for item in data_list])
    norm_vals = np.array([item["pushes_norm"] for item in data_list])
    # log_val = mean + std * norm_val  =>  fit lineal
    A = np.stack([np.ones_like(norm_vals), norm_vals], axis=1)
    result = np.linalg.lstsq(A, log_vals, rcond=None)
    mean_est, std_est = float(result[0][0]), float(result[0][1])
    return mean_est, std_est

rows, missing = [], []
bucket_errors_per_fold = {}   # {bucket: [mae_fold1, mae_fold2, ...]}

for fold in range(1, N_FOLDS + 1):
    model_path = os.path.join(RESULTS_DIR, f"final_regressor_fold{fold}.pt")
    test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")

    if not os.path.exists(model_path):
        print(f"  Warning: Fold {fold}: modelo no encontrado")
        missing.append(fold)
        continue
    if not os.path.exists(test_path):
        print(f"  Warning: Fold {fold}: datos de test no encontrados")
        missing.append(fold)
        continue

    # Cargar modelo (state_dict directo)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Cargar test set y recuperar stats de normalizacion
    data_list = torch.load(test_path, weights_only=False)
    p_mean, p_std = recover_norm_stats(data_list)

    dataset = RegressorTestDataset(data_list)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    # Inferencia
    all_pred_raw, all_gt_raw, all_buckets = [], [], []
    with torch.no_grad():
        for x, p_norm, p_raw, buckets in loader:
            x = x.to(device)
            pred_norm = model(x).cpu()
            # Desnormalizar: pred_real = expm1(pred_norm * std + mean)
            pred_real = torch.expm1(pred_norm * p_std + p_mean)
            all_pred_raw.extend(pred_real.numpy())
            all_gt_raw.extend(p_raw.numpy())
            all_buckets.extend(buckets)

    all_pred_raw = np.array(all_pred_raw)
    all_gt_raw   = np.array(all_gt_raw)
    all_buckets  = np.array(all_buckets)

    # Metricas globales
    abs_err  = np.abs(all_pred_raw - all_gt_raw)
    mae      = float(abs_err.mean())
    rmse     = float(np.sqrt((( all_pred_raw - all_gt_raw)**2).mean()))
    rho, _   = spearmanr(all_gt_raw, all_pred_raw)
    n_total  = len(all_gt_raw)

    print(f"  Fold {fold}:")
    print(f"    MAE={mae:.3f} | RMSE={rmse:.3f} | Spearman={rho:.4f}")
    print(f"    Pushes GT: min={all_gt_raw.min():.0f} max={all_gt_raw.max():.0f} "
          f"mean={all_gt_raw.mean():.1f} | N={n_total}")

    # Desglose por bucket de dificultad
    unique_buckets = sorted(set(all_buckets))
    print(f"    Desglose por bucket (MAE = |pred - gt| en pushes reales):")
    for bk in unique_buckets:
        mask    = (all_buckets == bk)
        bk_mae  = float(abs_err[mask].mean())
        bk_n    = int(mask.sum())
        bk_gt   = all_gt_raw[mask]
        print(f"      bucket={bk:>10s}: n={bk_n:5d} | MAE={bk_mae:6.2f} "
              f"| pushes=[{bk_gt.min():.0f}-{bk_gt.max():.0f}] mean={bk_gt.mean():.1f}")

        if bk not in bucket_errors_per_fold:
            bucket_errors_per_fold[bk] = []
        bucket_errors_per_fold[bk].append(bk_mae)

    print()
    rows.append({"fold": fold, "mae": mae, "rmse": rmse, "rho": rho})

# ── Resumen agregado ───────────────────────────────────────────────────────────
print("=" * 70)
if not rows:
    print("  Sin modelos. Copia los final_regressor_foldN.pt a surrogate_models/results/")
else:
    maes  = [r["mae"]  for r in rows]
    rmses = [r["rmse"] for r in rows]
    rhos  = [r["rho"]  for r in rows]

    print(f"  RESUMEN ({len(rows)}/{N_FOLDS} folds)")
    print("=" * 70)
    print(f"  MAE     : {np.mean(maes):.3f} +- {np.std(maes):.3f}  "
          f"[min={min(maes):.3f}, max={max(maes):.3f}]  (pushes reales)")
    print(f"  RMSE    : {np.mean(rmses):.3f} +- {np.std(rmses):.3f}")
    print(f"  Spearman: {np.mean(rhos):.4f} +- {np.std(rhos):.4f}")
    print()
    print(f"  Optuna vs CV: {OPTUNA_BEST_MAE:.2f} (val, sesgo optimista) -> {np.mean(maes):.3f} (test, estimacion honesta)")
    print(f"  Diferencia: {np.mean(maes) - OPTUNA_BEST_MAE:+.3f} pushes")
    print()

    if bucket_errors_per_fold:
        print("  Desglose por bucket (media +- std entre folds):")
        print(f"  {'Bucket':>12}  {'MAE Media':>10}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
        print(f"  {'-'*50}")
        for bk in sorted(bucket_errors_per_fold.keys()):
            vals = bucket_errors_per_fold[bk]
            print(f"  {bk:>12}  {np.mean(vals):10.3f}  "
                  f"+-{np.std(vals):5.3f}  "
                  f"[{min(vals):.3f} - {max(vals):.3f}]")

    if missing:
        print(f"\n  Folds faltantes: {missing}")
