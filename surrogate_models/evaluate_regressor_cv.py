"""
evaluate_regressor_cv.py
--------------------------
Evaluacion completa 5-fold CV del regresor final SE-ResNet.
Reporta: MAE, RMSE, Spearman global y por bucket, analisis de distribucion por fold.

Definiciones:
  - MAE:      Mean Absolute Error en pushes reales (escala original)
  - RMSE:     Root Mean Squared Error en pushes reales
  - Spearman: Correlacion de ranking entre predicciones y ground truth
  - Bucket:   Agrupacion por dificultad (pushes reales) del puzzle

Interpretacion correcta del error:
  - El error ABSOLUTO crece con la dificultad a tasa sub-lineal.
  - El error RELATIVO (MAE/mean_pushes) es mayor en la zona facil (efecto de
    dividir por valores pequenos), no es evidencia de mejor desempeno en la zona dificil.

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
SPEARMAN_MIN_N = 30   # minimo de muestras por bucket para reportar Spearman
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  EVALUACION CROSS-VALIDATION: REGRESOR (SE-ResNet)")
print("=" * 70)
print(f"  Dispositivo: {device}")
print(f"  Ref. Optuna (val, sesgo optimista): MAE = {OPTUNA_BEST_MAE:.2f} pushes")
print()
print("  Metrica principal: MAE en pushes reales (escala original)")
print("  Nota: error absoluto crece con dificultad a tasa sub-lineal.")
print("        error relativo mayor en zona facil (artefacto de dividir por")
print("        valores pequenos, no ventaja real en zona dificil).")
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
    tensors = torch.stack([b[0] for b in batch])
    p_norm  = torch.stack([b[1] for b in batch])
    p_raw   = torch.stack([b[2] for b in batch])
    buckets = [b[3] for b in batch]
    return tensors, p_norm, p_raw, buckets

def recover_norm_stats(data_list):
    """Recupera mean y std de normalizacion via regresion lineal."""
    log_vals  = np.array([np.log1p(item["pushes_raw"])  for item in data_list])
    norm_vals = np.array([item["pushes_norm"] for item in data_list])
    A = np.stack([np.ones_like(norm_vals), norm_vals], axis=1)
    result = np.linalg.lstsq(A, log_vals, rcond=None)
    return float(result[0][0]), float(result[0][1])

rows, missing = [], []
# Acumuladores por bucket para resumen con std
bucket_maes_per_fold = {}
bucket_rhos_per_fold = {}
# Para analisis de distribucion por fold
fold_dist = {}

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

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_list = torch.load(test_path, weights_only=False)
    p_mean, p_std = recover_norm_stats(data_list)

    dataset = RegressorTestDataset(data_list)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    all_pred_raw, all_gt_raw, all_buckets = [], [], []
    with torch.no_grad():
        for x, p_norm, p_raw, buckets in loader:
            x = x.to(device)
            pred_norm = model(x).cpu()
            pred_real = torch.expm1(pred_norm * p_std + p_mean)
            all_pred_raw.extend(pred_real.numpy())
            all_gt_raw.extend(p_raw.numpy())
            all_buckets.extend(buckets)

    all_pred_raw = np.array(all_pred_raw)
    all_gt_raw   = np.array(all_gt_raw)
    all_buckets  = np.array(all_buckets)

    abs_err = np.abs(all_pred_raw - all_gt_raw)
    mae     = float(abs_err.mean())
    rmse    = float(np.sqrt(((all_pred_raw - all_gt_raw)**2).mean()))
    rho, _  = spearmanr(all_gt_raw, all_pred_raw)
    n_total = len(all_gt_raw)

    # Distribucion por fold para analisis de sesgo
    pct_hard = float((all_gt_raw >= 71).sum() / n_total * 100)
    pct_very_hard = float((all_gt_raw >= 91).sum() / n_total * 100)
    fold_dist[fold] = {
        "mean_pushes": float(all_gt_raw.mean()),
        "median_pushes": float(np.median(all_gt_raw)),
        "pct_71plus": pct_hard,
        "pct_91plus": pct_very_hard,
        "n": n_total
    }

    print(f"  Fold {fold}:")
    print(f"    MAE={mae:.3f} | RMSE={rmse:.3f} | Spearman(global)={rho:.4f}")
    print(f"    Pushes GT: min={all_gt_raw.min():.0f} max={all_gt_raw.max():.0f} "
          f"mean={all_gt_raw.mean():.1f} | N={n_total}")

    unique_buckets = sorted(set(all_buckets))
    print(f"    Desglose por bucket (MAE, error relativo, Spearman intra-bucket):")
    print(f"      {'Bucket':>12}  {'n':>5}  {'MAE':>6}  {'Err%':>5}  {'Spearman':>9}")
    for bk in unique_buckets:
        mask    = (all_buckets == bk)
        bk_mae  = float(abs_err[mask].mean())
        bk_n    = int(mask.sum())
        bk_gt   = all_gt_raw[mask]
        bk_pred = all_pred_raw[mask]
        bk_mean = float(bk_gt.mean())
        rel_err = bk_mae / bk_mean * 100  # error relativo (artefacto en zona facil)

        # Spearman intra-bucket (solo si hay suficientes muestras con varianza)
        if bk_n >= SPEARMAN_MIN_N and len(np.unique(bk_gt)) > 1:
            bk_rho, _ = spearmanr(bk_gt, bk_pred)
            rho_str = f"{bk_rho:+.3f}"
        else:
            bk_rho  = float("nan")
            rho_str = "  n/a  "

        print(f"      {bk:>12}  {bk_n:5d}  {bk_mae:6.2f}  {rel_err:4.0f}%  {rho_str:>9}")

        if bk not in bucket_maes_per_fold:
            bucket_maes_per_fold[bk] = []
            bucket_rhos_per_fold[bk] = []
        bucket_maes_per_fold[bk].append(bk_mae)
        if not np.isnan(bk_rho):
            bucket_rhos_per_fold[bk].append(bk_rho)

    print()
    rows.append({"fold": fold, "mae": mae, "rmse": rmse, "rho": rho})

# ── Analisis de distribucion: anomalia Fold 5 ─────────────────────────────────
print("=" * 70)
print("  ANALISIS DE DISTRIBUCION POR FOLD (anomalia Fold 2 y 5)")
print("=" * 70)
print(f"  {'Fold':>5}  {'mean_push':>10}  {'median':>7}  {'%>=71':>6}  {'%>=91':>6}  {'MAE':>6}  {'Spearman':>9}")
for fold in sorted(fold_dist.keys()):
    d   = fold_dist[fold]
    r   = next((r for r in rows if r["fold"] == fold), None)
    mae_str = f"{r['mae']:.3f}" if r else "  n/a"
    rho_str = f"{r['rho']:.4f}" if r else "  n/a"
    marker = " <-- atipico" if fold in (2, 5) else ""
    print(f"  {fold:>5}  {d['mean_pushes']:10.1f}  {d['median_pushes']:7.1f}  "
          f"{d['pct_71plus']:5.1f}%  {d['pct_91plus']:5.1f}%  "
          f"{mae_str:>6}  {rho_str:>9}{marker}")
print()
print("  Interpretacion:")
print("  - Si Fold 2 y 5 tienen mayor %>=71 o %>=91 que los demas,")
print("    la anomalia se explica por mayor concentracion de casos complejos")
print("    (sesgo en la particion GroupKFold, no fallo del modelo).")
print("  - Consistente con anomalia del clasificador (Fold 5: CORRAL Error=19.6%,")
print("    threshold=0.80 vs 0.89-0.90 del resto).")

# ── Resumen agregado ───────────────────────────────────────────────────────────
print()
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

    if bucket_maes_per_fold:
        print("  Desglose por bucket (media +- std entre folds, MAE y Spearman):")
        print(f"  {'Bucket':>12}  {'MAE media':>10}  {'+-std':>6}  {'Rho media':>10}  {'+-std':>6}")
        print(f"  {'-'*58}")
        for bk in sorted(bucket_maes_per_fold.keys()):
            mvals = bucket_maes_per_fold[bk]
            rvals = bucket_rhos_per_fold.get(bk, [])
            rho_str = f"{np.mean(rvals):.3f} +-{np.std(rvals):.3f}" if rvals else "    n/a     "
            print(f"  {bk:>12}  {np.mean(mvals):10.3f}  "
                  f"+-{np.std(mvals):5.3f}  "
                  f"  {rho_str}")

    if missing:
        print(f"\n  Folds faltantes: {missing}")
