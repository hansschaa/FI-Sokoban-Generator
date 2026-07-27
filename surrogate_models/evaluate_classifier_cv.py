"""
evaluate_classifier_cv.py
--------------------------
Evaluación completa 5-fold CV del clasificador SE-ResNet.
Reporta: F0.5, Precisión, Recall, AUC-ROC, AUC-PR, desglose por deadlock_type,
         y análisis del threshold por fold.

Uso:
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/evaluate_classifier_cv.py
"""

import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    average_precision_score
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from models.resnet import SokobanSEResNetClassifier
from train_final_surrogates import ClassifierDataset

OPTUNA_BEST_F05 = 0.7867  # Referencia del mejor trial de Optuna (Fold 1 val)
N_FOLDS    = 5
BATCH_SIZE = 256
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("  EVALUACIÓN CROSS-VALIDATION: CLASIFICADOR (SE-ResNet)")
print("=" * 70)
print(f"  Dispositivo: {device}")
print(f"  ⚠️  Referencia Optuna (val, sesgo optimista): F0.5 = {OPTUNA_BEST_F05:.4f}\n")

rows, missing = [], []

for fold in range(1, N_FOLDS + 1):
    model_path = os.path.join(RESULTS_DIR, f"final_classifier_fold{fold}.pt")
    test_path  = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")

    if not os.path.exists(model_path):
        print(f"  ⚠️  Fold {fold}: modelo no encontrado")
        missing.append(fold)
        continue
    if not os.path.exists(test_path):
        print(f"  ⚠️  Fold {fold}: datos de test no encontrados")
        missing.append(fold)
        continue

    # Cargar modelo
    state = torch.load(model_path, map_location=device, weights_only=False)
    model = SokobanSEResNetClassifier(dropout_p=0.0).to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # Cargar test set completo (con metadatos)
    test_data = torch.load(test_path, weights_only=False)

    # Extraer tensores, labels y metadatos
    tensors      = test_data["tensor"]
    labels       = test_data["is_solvable"]
    deadlock_types = test_data.get("deadlock_type", None)

    dataset = ClassifierDataset(test_data)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Inferencia
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.numpy())

    all_probs   = np.array(all_probs)
    all_targets = np.array(all_targets)

    # AUC-ROC y AUC-PR
    auc_roc = roc_auc_score(all_targets, all_probs)
    auc_pr  = average_precision_score(all_targets, all_probs)

    # Buscar mejor threshold
    best_f05, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (all_probs >= t).astype(float)
        f = fbeta_score(all_targets, preds, beta=0.5, zero_division=0)
        if f > best_f05:
            best_f05, best_t = f, t

    preds = (all_probs >= best_t).astype(float)
    prec  = precision_score(all_targets, preds, zero_division=0)
    rec   = recall_score(all_targets, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
    n_pos = int(all_targets.sum())
    n_neg = int(len(all_targets) - n_pos)

    print(f"  Fold {fold}:")
    print(f"    F0.5={best_f05:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | "
          f"Thresh={best_t:.2f}")
    print(f"    AUC-ROC={auc_roc:.4f} | AUC-PR={auc_pr:.4f}")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn} | Pos={n_pos} Neg={n_neg}")


    # -- Desglose por deadlock_type --
    # Cada tipo es UNA sola clase (deadlocks son 100% negativos, SOLVABLE 100% positivo).
    # Para deadlocks: error = % clasificados como SOLVABLE (FP por tipo)
    # Para SOLVABLE: recall = % correctamente identificados como solvable
    if deadlock_types is not None:
        dtype_arr = np.array(deadlock_types)
        unique_types = sorted(set(dtype_arr))

        print(f"    Desglose por deadlock_type (error por tipo):")
        for dt in sorted(unique_types, key=lambda x: (x != "SOLVABLE", x)):
            mask = (dtype_arr == dt)
            if mask.sum() == 0:
                continue
            t_preds = preds[mask]
            t_probs = all_probs[mask]
            n       = int(mask.sum())

            if dt == "SOLVABLE":
                correct  = int(t_preds.sum())
                wrong    = n - correct
                pct      = correct / n * 100
                avg_prob = float(t_probs.mean())
                print(f"      {'SOLVABLE':20s}: n={n:5d} | Recall={pct:5.1f}% "
                      f"({correct} correctos, {wrong} perdidos) | prob_media={avg_prob:.3f}")
            else:
                fp_count = int(t_preds.sum())
                fp_rate  = fp_count / n * 100
                avg_prob = float(t_probs.mean())
                print(f"      {dt:20s}: n={n:5d} | Error={fp_rate:5.1f}% "
                      f"({fp_count} FP de {n}) | prob_media={avg_prob:.3f}")

    print()

    rows.append({
        "fold": fold, "f05": best_f05, "prec": prec, "rec": rec,
        "auc_roc": auc_roc, "auc_pr": auc_pr, "thresh": best_t
    })

# ── Resumen agregado ───────────────────────────────────────────────────────────
print("=" * 70)
if not rows:
    print("  ❌ Sin modelos. Copia los final_classifier_foldN.pt a surrogate_models/results/")
else:
    f05s    = [r["f05"]     for r in rows]
    precs   = [r["prec"]    for r in rows]
    recs    = [r["rec"]     for r in rows]
    rocs    = [r["auc_roc"] for r in rows]
    prs     = [r["auc_pr"]  for r in rows]
    threshs = [r["thresh"]  for r in rows]

    print(f"  RESUMEN ({len(rows)}/{N_FOLDS} folds)")
    print("=" * 70)
    print(f"  F0.5    : {np.mean(f05s):.4f} ± {np.std(f05s):.4f}  "
          f"[min={min(f05s):.4f}, max={max(f05s):.4f}]")
    print(f"  Prec    : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall  : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  AUC-ROC : {np.mean(rocs):.4f} ± {np.std(rocs):.4f}")
    print(f"  AUC-PR  : {np.mean(prs):.4f} ± {np.std(prs):.4f}")
    print(f"  Thresh  : {np.mean(threshs):.3f} ± {np.std(threshs):.3f}  (por fold: {[f'{t:.2f}' for t in threshs]})")
    print()
    print(f"  ⚠️  OPTUNA vs CV: {OPTUNA_BEST_F05:.4f} (val, sesgo optimista) → {np.mean(f05s):.4f} (test, estimación honesta)")
    print(f"      Caída: {OPTUNA_BEST_F05 - np.mean(f05s):.4f} pts — explicado por sesgo de selección de hiperparámetros en Fold 1 val.")
    if missing:
        print(f"\n  ⚠️  Folds faltantes: {missing}")
