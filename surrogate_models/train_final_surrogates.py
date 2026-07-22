"""
train_final_surrogates.py
-------------------------
Entrena los modelos finales (5-Fold Cross Validation) para el Regresor y el Clasificador
utilizando los hiperparámetros campeones encontrados por Optuna.

Genera la tabla académica definitiva de resultados (Media ± Desviación Estándar)
y guarda los 5 checkpoints de modelo (.pt) para cada arquitectura en results/.
"""

import sys, os, json, copy, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

from models.resnet import SokobanResNetRegressor, MultiHeadRegressorLoss, SokobanResNetClassifier, ClassifierLoss

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
class RegressorDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['tensor'].float(),
            torch.tensor(item['pushes_norm'], dtype=torch.float32),
            torch.tensor(item['branch_norm'], dtype=torch.float32),
            torch.tensor(item['pushes_raw'],  dtype=torch.float32),
            torch.tensor(item['branch_raw'],  dtype=torch.float32),
        )

class ClassifierDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['tensor'].float(),
            torch.tensor(item['is_solvable'], dtype=torch.float32)
        )

# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO REGRESOR (5 FOLDS)
# ─────────────────────────────────────────────────────────────────────────────
def train_final_regressor():
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    if not os.path.exists(hparams_path):
        print("❌ Error: No se encontró best_hparams.json")
        return None

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p = cfg["dropout_p"]
    w_branch = cfg["w_branch"]
    batch_size = int(cfg["batch_size"])

    print("\n" + "="*65)
    print("  ENTRENAMIENTO FINAL 5-FOLD: SURROGATE REGRESSOR")
    print("="*65)
    print(f"  Hiperparámetros: lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, w_branch={w_branch:.2f}, bs={batch_size}\n")

    fold_maes_pushes = []
    
    for fold in range(1, 6):
        train_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_train.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")

        if not os.path.exists(train_path):
            print(f"⚠️ Saltando Fold {fold}: no se encontraron los datos.")
            continue

        train_data = torch.load(train_path, weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)
        stats      = torch.load(stats_path, weights_only=False)
        p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]

        train_loader = DataLoader(RegressorDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(RegressorDataset(test_data),  batch_size=256, shuffle=False, num_workers=0)

        model = SokobanResNetRegressor(dropout_p=dropout_p).to(device)
        criterion = MultiHeadRegressorLoss(w_pushes=1.0, w_branch=w_branch)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        best_mae = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        for epoch in range(1, 51):
            model.train()
            for tensors, p_norm, b_norm, _, _ in train_loader:
                tensors, p_norm, b_norm = tensors.to(device), p_norm.to(device), b_norm.to(device)
                optimizer.zero_grad()
                p_pred, b_pred = model(tensors)
                loss, _ = criterion(p_pred, p_norm, b_pred, b_norm)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # Eval
            model.eval()
            total_mae, n = 0.0, 0
            with torch.no_grad():
                for tensors, _, _, p_raw, _ in test_loader:
                    tensors = tensors.to(device)
                    p_pred, _ = model(tensors)
                    p_desnorm = p_pred.cpu() * p_std + p_mean
                    total_mae += torch.abs(p_desnorm - p_raw).sum().item()
                    n += len(p_raw)
            
            mae = total_mae / n
            scheduler.step(mae)

            if mae < best_mae:
                best_mae = mae
                best_weights = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= 10:
                    break

        fold_maes_pushes.append(best_mae)
        save_path = os.path.join(RESULTS_DIR, f"final_regressor_fold{fold}.pt")
        torch.save(best_weights, save_path)
        print(f"  Fold {fold} finalizado | Mejor MAE Pushes: {best_mae:.2f} empujes | Guardado en {os.path.basename(save_path)}")

    mean_mae = np.mean(fold_maes_pushes)
    std_mae  = np.std(fold_maes_pushes)
    print(f"\n  🏆 REGRESOR (5-FOLD CV): MAE Pushes = {mean_mae:.2f} ± {std_mae:.2f} empujes")
    return mean_mae, std_mae

# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO CLASIFICADOR (5 FOLDS)
# ─────────────────────────────────────────────────────────────────────────────
def train_final_classifier():
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams_classifier.json")
    if not os.path.exists(hparams_path):
        print("❌ Error: No se encontró best_hparams_classifier.json")
        return None

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p = cfg["dropout_p"]
    pos_weight = cfg["pos_weight"]
    batch_size = int(cfg["batch_size"])

    print("\n" + "="*65)
    print("  ENTRENAMIENTO FINAL 5-FOLD: SURROGATE CLASSIFIER")
    print("="*65)
    print(f"  Hiperparámetros: lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, pos_w={pos_weight:.2f}, bs={batch_size}\n")

    fold_accs, fold_precs, fold_recs, fold_f1s, fold_f05s, fold_aucs = [], [], [], [], [], []

    for fold in range(1, 6):
        train_path = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_train.pt")
        test_path  = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")

        if not os.path.exists(train_path):
            print(f"⚠️ Saltando Fold {fold}: no se encontraron los datos.")
            continue

        train_data = torch.load(train_path, weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)

        train_loader = DataLoader(ClassifierDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(ClassifierDataset(test_data),  batch_size=256, shuffle=False, num_workers=0)

        model = SokobanResNetClassifier(dropout_p=dropout_p).to(device)
        criterion = ClassifierLoss(pos_weight_val=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_f05 = 0.0
        best_metrics = {}
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        for epoch in range(1, 41):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            # Eval
            model.eval()
            all_preds, all_probs, all_targets = [], [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    logits = model(x)
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y.numpy())

            acc  = accuracy_score(all_targets, all_preds)
            prec = precision_score(all_targets, all_preds, zero_division=0)
            rec  = recall_score(all_targets, all_preds, zero_division=0)
            f1   = f1_score(all_targets, all_preds, zero_division=0)
            f05  = fbeta_score(all_targets, all_preds, beta=0.5, zero_division=0)
            try: auc = roc_auc_score(all_targets, all_probs)
            except ValueError: auc = 0.0

            scheduler.step(f05)

            if f05 > best_f05:
                best_f05 = f05
                best_metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "f05": f05, "auc": auc}
                best_weights = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= 8:
                    break

        fold_accs.append(best_metrics["acc"])
        fold_precs.append(best_metrics["prec"])
        fold_recs.append(best_metrics["rec"])
        fold_f1s.append(best_metrics["f1"])
        fold_f05s.append(best_metrics["f05"])
        fold_aucs.append(best_metrics["auc"])

        save_path = os.path.join(RESULTS_DIR, f"final_classifier_fold{fold}.pt")
        torch.save(best_weights, save_path)
        print(f"  Fold {fold} finalizado | Acc={best_metrics['acc']:.3f} | Prec={best_metrics['prec']:.3f} | "
              f"Rec={best_metrics['rec']:.3f} | F1={best_metrics['f1']:.3f} | F0.5={best_metrics['f05']:.3f} | AUC={best_metrics['auc']:.3f}")

    print("\n" + "="*65)
    print("  🏆 CLASIFICADOR (5-FOLD CV): TABLA ACADÉMICA FINAL")
    print("="*65)
    print(f"  Accuracy  : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"  Precision : {np.mean(fold_precs):.4f} ± {np.std(fold_precs):.4f}")
    print(f"  Recall    : {np.mean(fold_recs):.4f} ± {np.std(fold_recs):.4f}")
    print(f"  F1-Score  : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  F0.5-Score: {np.mean(fold_f05s):.4f} ± {np.std(fold_f05s):.4f}")
    print(f"  AUC-ROC   : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

if __name__ == "__main__":
    train_final_regressor()
    train_final_classifier()
