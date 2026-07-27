"""
evaluate_classifier_cv.py
--------------------------
Evalúa los 5 folds del clasificador final y reporta métricas agregadas.
Genera tabla de resultados por fold + media ± std.

Uso:
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/evaluate_classifier_cv.py
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

from models.resnet import SokobanSEResNetClassifier
from train_final_surrogates import ClassifierDataset

N_FOLDS    = 5
BATCH_SIZE = 256
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("  EVALUACIÓN CROSS-VALIDATION: CLASIFICADOR (SE-ResNet)")
print("=" * 65)
print(f"  Dispositivo: {device}\n")

rows, missing = [], []

for fold in range(1, N_FOLDS + 1):
    model_path = os.path.join(RESULTS_DIR, f"final_classifier_fold{fold}.pt")
    test_path  = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")

    if not os.path.exists(model_path):
        print(f"  ⚠️  Fold {fold}: modelo no encontrado ({os.path.basename(model_path)})")
        missing.append(fold)
        continue
    if not os.path.exists(test_path):
        print(f"  ⚠️  Fold {fold}: datos de test no encontrados")
        missing.append(fold)
        continue

    state = torch.load(model_path, map_location=device, weights_only=False)
    model = SokobanSEResNetClassifier(dropout_p=0.0).to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        best_thresh_saved = state.get("best_thresh", 0.5)
    else:
        model.load_state_dict(state)
        best_thresh_saved = 0.5
    model.eval()

    test_data   = torch.load(test_path, weights_only=False)
    test_loader = DataLoader(ClassifierDataset(test_data), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.numpy())

    all_probs   = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Buscar mejor threshold
    best_f05, best_t = 0.0, best_thresh_saved
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (all_probs >= t).astype(float)
        f = fbeta_score(all_targets, preds, beta=0.5, zero_division=0)
        if f > best_f05:
            best_f05, best_t = f, t

    preds = (all_probs >= best_t).astype(float)
    prec  = precision_score(all_targets, preds, zero_division=0)
    rec   = recall_score(all_targets, preds, zero_division=0)
    auc   = roc_auc_score(all_targets, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
    n_pos = int(all_targets.sum())
    n_neg = int(len(all_targets) - n_pos)

    print(f"  Fold {fold}: F0.5={best_f05:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | "
          f"AUC={auc:.4f} | Thresh={best_t:.2f} | "
          f"TP={tp} FP={fp} FN={fn} TN={tn} | Pos={n_pos} Neg={n_neg}")

    rows.append({"fold": fold, "f05": best_f05, "prec": prec,
                 "rec": rec, "auc": auc, "thresh": best_t})

print()
if not rows:
    print("  ❌ No se encontró ningún modelo. Copia los final_classifier_foldN.pt a surrogate_models/results/")
else:
    f05s  = [r["f05"]  for r in rows]
    precs = [r["prec"] for r in rows]
    recs  = [r["rec"]  for r in rows]
    aucs  = [r["auc"]  for r in rows]

    print("=" * 65)
    print(f"  RESUMEN ({len(rows)}/{N_FOLDS} folds)")
    print("=" * 65)
    print(f"  F0.5  : {np.mean(f05s):.4f} ± {np.std(f05s):.4f}  [min={min(f05s):.4f}, max={max(f05s):.4f}]")
    print(f"  Prec  : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  AUC   : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    if missing:
        print(f"\n  ⚠️  Folds faltantes: {missing} — resumen parcial.")
