"""
train_final_surrogates.py
-------------------------
Entrena los modelos finales (5-Fold Cross Validation) para el Regresor y/o Clasificador
utilizando los hiperparámetros campeones de Optuna.

Soporta paralelización entre computadoras:
    --model regressor   -> Entrena solo el Regresor
    --model classifier  -> Entrena solo el Clasificador
    --folds 1,2         -> Entrena solo los Folds especificados

Muestra progreso en tiempo real época por época con tiempo estimado (ETA).
"""

import sys, os, json, copy, time, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

from models.resnet import SokobanSEResNetRegressor, MultiHeadRegressorLoss, SokobanSEResNetClassifier, ClassifierLoss

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
            torch.tensor(item.get('weight', 1.0), dtype=torch.float32),
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
# REGRESOR
# ─────────────────────────────────────────────────────────────────────────────
def train_regressor(folds_to_run):
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

    print("\n" + "="*65)
    print("  ENTRENAMIENTO FINAL: REGRESOR (SINGLE-HEAD SERESNET)")
    print("="*65)
    print(f"  Dispositivo  : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Hiperparámetros: lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, bs={batch_size}\n")

    fold_maes = []

    for fold in folds_to_run:
        train_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_train.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")

        if not os.path.exists(train_path):
            print(f"⚠️ Saltando Fold {fold}: No existe {train_path}")
            continue

        print(f"\n[{'─'*40}]")
        print(f"  INICIANDO FOLD {fold}/5")
        print(f"[{'─'*40}]")

        train_data = torch.load(train_path, weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)
        stats      = torch.load(stats_path, weights_only=False)
        p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]

        train_loader = DataLoader(RegressorDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(RegressorDataset(test_data),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

        model     = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
        criterion = nn.HuberLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_mae     = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        for epoch in range(1, 51):
            t0 = time.time()
            model.train()
            train_loss = 0.0

            for tensors, p_norm, _, _, _, weights in train_loader:
                tensors, p_norm, weights = tensors.to(device), p_norm.to(device), weights.to(device)
                optimizer.zero_grad()
                p_pred = model(tensors)
                
                loss_p = criterion(p_pred, p_norm)
                loss = (loss_p * weights).mean()
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            total_mae, n = 0.0, 0
            with torch.no_grad():
                for tensors, _, _, p_raw, _, _ in test_loader:
                    tensors = tensors.to(device)
                    p_pred = model(tensors)
                    p_desnorm = p_pred.cpu() * p_std + p_mean
                    p_desnorm_real = torch.expm1(p_desnorm)
                    total_mae += torch.abs(p_desnorm_real - p_raw).sum().item()
                    n += len(p_raw)

            mae = total_mae / n
            scheduler.step()

            elapsed = time.time() - t0
            tag = ""
            if mae < best_mae:
                best_mae = mae
                best_weights = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                tag = " ★ (Nuevo récord)"
            else:
                patience_ctr += 1

            print(f"  Ep {epoch:02d} | T: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | MAE Test: {mae:.2f} empujes{tag}")

            if patience_ctr >= 10:
                print(f"  🛑 Early Stopping en época {epoch}.")
                break

        fold_maes.append(best_mae)
        save_path = os.path.join(RESULTS_DIR, f"final_regressor_fold{fold}.pt")
        torch.save(best_weights, save_path)
        print(f"  ✅ Fold {fold} guardado en {os.path.basename(save_path)} | Mejor MAE: {best_mae:.2f}")

    if len(fold_maes) > 1:
        print(f"\n  🏆 REGRESOR FINAL (5-FOLD CV): MAE Pushes = {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} empujes")

# ─────────────────────────────────────────────────────────────────────────────
# CLASIFICADOR
# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(folds_to_run):
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams_classifier.json")
    if not os.path.exists(hparams_path):
        print("❌ Error: No se encontró best_hparams_classifier.json")
        return

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr           = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p    = cfg["dropout_p"]
    pos_weight   = cfg["pos_weight"]
    batch_size   = int(cfg["batch_size"])

    print("\n" + "="*65)
    print("  ENTRENAMIENTO FINAL: CLASIFICADOR (RESNET)")
    print("="*65)
    print(f"  Dispositivo  : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Hiperparámetros: lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, pos_w={pos_weight:.2f}, bs={batch_size}\n")

    fold_accs, fold_precs, fold_recs, fold_f1s, fold_f05s, fold_aucs = [], [], [], [], [], []

    for fold in folds_to_run:
        train_path = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_train.pt")
        test_path  = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")

        if not os.path.exists(train_path):
            print(f"⚠️ Saltando Fold {fold}: No existe {train_path}")
            continue

        print(f"\n[{'─'*40}]")
        print(f"  INICIANDO FOLD {fold}/5")
        print(f"[{'─'*40}]")

        train_data = torch.load(train_path, weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)

        train_loader = DataLoader(ClassifierDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(ClassifierDataset(test_data),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

        model     = SokobanSEResNetClassifier(dropout_p=dropout_p).to(device)
        criterion = ClassifierLoss(pos_weight_val=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_f05     = 0.0
        best_metrics = {}
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        for epoch in range(1, 41):
            t0 = time.time()
            model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

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

            scheduler.step()

            elapsed = time.time() - t0
            tag = ""
            if f05 > best_f05:
                best_f05 = f05
                best_metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "f05": f05, "auc": auc}
                best_weights = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                tag = " ★ (Nuevo récord)"
            else:
                patience_ctr += 1

            print(f"  Ep {epoch:02d} | T: {elapsed:.1f}s | Loss: {train_loss:.4f} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | F0.5: {f05:.3f}{tag}")

            if patience_ctr >= 8:
                print(f"  🛑 Early Stopping en época {epoch}.")
                break

        fold_accs.append(best_metrics["acc"])
        fold_precs.append(best_metrics["prec"])
        fold_recs.append(best_metrics["rec"])
        fold_f1s.append(best_metrics["f1"])
        fold_f05s.append(best_metrics["f05"])
        fold_aucs.append(best_metrics["auc"])

        save_path = os.path.join(RESULTS_DIR, f"final_classifier_fold{fold}.pt")
        torch.save(best_weights, save_path)
        print(f"  ✅ Fold {fold} guardado en {os.path.basename(save_path)} | Mejor F0.5: {best_metrics['f05']:.4f}")

    if len(fold_f05s) > 1:
        print("\n" + "="*65)
        print("  🏆 CLASIFICADOR FINAL (5-FOLD CV): TABLA ACADÉMICA")
        print("="*65)
        print(f"  Accuracy  : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"  Precision : {np.mean(fold_precs):.4f} ± {np.std(fold_precs):.4f}")
        print(f"  Recall    : {np.mean(fold_recs):.4f} ± {np.std(fold_recs):.4f}")
        print(f"  F1-Score  : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        print(f"  F0.5-Score: {np.mean(fold_f05s):.4f} ± {np.std(fold_f05s):.4f}")
        print(f"  AUC-ROC   : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento Final 5-Fold para Modelos Subrogados")
    parser.add_argument("--model", type=str, default="all", choices=["all", "regressor", "classifier"],
                        help="Modelo a entrenar: regressor, classifier o all (ambos)")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5",
                        help="Folds a ejecutar separados por coma (ej. 1,2,3 o 1)")

    args = parser.parse_args()
    folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    if args.model in ["all", "regressor"]:
        train_regressor(folds_to_run)

    if args.model in ["all", "classifier"]:
        train_classifier(folds_to_run)

if __name__ == "__main__":
    main()
