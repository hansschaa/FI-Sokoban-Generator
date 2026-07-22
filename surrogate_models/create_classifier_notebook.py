"""
create_classifier_notebook.py
-----------------------------
Genera el Jupyter Notebook para entrenar el Categorizador (SokobanResNetClassifier).
Carga automáticamente los mejores hiperparámetros de best_hparams_classifier.json si existen.
"""

import nbformat as nbf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

nb = nbf.v4.new_notebook()

def md(text): return nbf.v4.new_markdown_cell(text)
def cell(code): return nbf.v4.new_code_cell(code)

cells = []

# ── SECCIÓN 1: Imports y Configuración ────────────────────────────────────
cells.append(md("# Entrenamiento de Surrogate Model: Clasificador (Factible vs Deadlock)\n\n"
                "Usa los folds generados por `prepare_classifier.py` y los hiperparámetros encontrados por Optuna.\n"
                "Optimiza Weighted Binary Cross-Entropy para manejar el desbalance."))

cells.append(cell("""\
import os, sys, copy, time, json
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

from models.resnet import SokobanResNetClassifier, ClassifierLoss

RESULTS_DIR = '../results'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar mejores hiperparámetros de Optuna
hparams_path = f"{RESULTS_DIR}/best_hparams_classifier.json"
if os.path.exists(hparams_path):
    with open(hparams_path, "r") as f:
        best_cfg = json.load(f)["params"]
    print("✅ Hiperparámetros de Optuna cargados:")
    for k, v in best_cfg.items():
        print(f"   {k}: {v}")
else:
    print("⚠️ No se encontró best_hparams_classifier.json. Usando valores por defecto.")
    best_cfg = {"lr": 0.001, "weight_decay": 1e-5, "dropout_p": 0.4, "pos_weight": 1.0, "batch_size": 128}
"""))

# ── SECCIÓN 2: Dataset y Dataloaders ──────────────────────────────────────
cells.append(md("## 1. Carga de Datos"))
cells.append(cell("""\
class FoldDataset(Dataset):
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

def get_fold_loaders(fold_idx, batch_size=128):
    train_data = torch.load(f'{RESULTS_DIR}/classifier_fold{fold_idx}_train.pt', weights_only=False)
    test_data  = torch.load(f'{RESULTS_DIR}/classifier_fold{fold_idx}_test.pt',  weights_only=False)
    
    labels = [d['is_solvable'] for d in train_data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    train_loader = DataLoader(FoldDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(FoldDataset(test_data),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Fold {fold_idx}: Train={len(train_data):,} | Test={len(test_data):,}")
    print(f"Distribución (Train): Solubles={n_pos:,} | Deadlocks={n_neg:,}")
    
    return train_loader, test_loader
"""))

# ── SECCIÓN 3: Entrenamiento ──────────────────────────────────────────────
cells.append(md("## 2. Bucle de Entrenamiento"))
cells.append(cell("""\
def train_fold(fold_idx, epochs=30, patience=8):
    print(f"\\n{'='*55}\\n FOLD {fold_idx}/5\\n{'='*55}")
    
    lr = best_cfg.get("lr", 1e-3)
    weight_decay = best_cfg.get("weight_decay", 1e-5)
    dropout_p = best_cfg.get("dropout_p", 0.4)
    pos_weight = best_cfg.get("pos_weight", 1.0)
    batch_size = int(best_cfg.get("batch_size", 128))
    
    train_loader, test_loader = get_fold_loaders(fold_idx, batch_size=batch_size)
    
    model = SokobanResNetClassifier(dropout_p=dropout_p).to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f_beta = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    
    history = {'loss': [], 'val_loss': [], 'val_f1': [], 'val_f05': [], 'val_auc': []}
    
    for epoch in range(1, epochs + 1):
        # TRAIN
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
        
        # EVAL
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        val_loss /= len(test_loader)
        
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        f05 = fbeta_score(all_targets, all_preds, beta=0.5, zero_division=0)
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.0
            
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(f1)
        history['val_f05'].append(f05)
        history['val_auc'].append(auc)
        
        scheduler.step(f05)
        
        tag = ""
        if f05 > best_f_beta:
            best_f_beta = f05
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = " ★"
        else:
            patience_ctr += 1
            
        print(f"Ep {epoch:03d} | L: {train_loss:.3f} | vL: {val_loss:.3f} | "
              f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | F0.5: {f05:.3f} | AUC: {auc:.3f}{tag}")
              
        if patience_ctr >= patience:
            print(f"🛑 Early Stopping en época {epoch}.")
            break
            
    out = f"{RESULTS_DIR}/classifier_fold{fold_idx}_model.pt"
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), out)
    print(f"✅ Mejor modelo guardado en {out} con F0.5={best_f_beta:.4f}")
    
    return history
"""))

cells.append(md("## 3. Ejecución K-Fold"))
cells.append(cell("""\
all_histories = []
for fold in range(1, 6):
    h = train_fold(fold)
    all_histories.append(h)
"""))

nb['cells'] = cells
out_path = os.path.join(NOTEBOOKS_DIR, "train_classifier.ipynb")
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Notebook del Clasificador generado en: {out_path}")
