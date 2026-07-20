"""
create_classifier_notebook.py
-----------------------------
Genera el Jupyter Notebook para entrenar el Categorizador (SokobanResNetClassifier).
Incluye métricas de clasificación (Accuracy, Precision, Recall, F1, ROC-AUC) 
y maneja el desbalance con pos_weight.
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
                "Usa los folds generados por `prepare_classifier.py`.\n"
                "Optimiza Weighted Binary Cross-Entropy para manejar el desbalance."))

cells.append(cell("""\
import os, sys, copy, time
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.resnet import SokobanResNetClassifier, ClassifierLoss

# Configuración
RESULTS_DIR = '../results'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
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
            torch.tensor(item['label'], dtype=torch.float32)
        )

def get_fold_loaders(fold_idx, batch_size=128):
    train_data = torch.load(f'{RESULTS_DIR}/classifier_fold{fold_idx}_train.pt', weights_only=False)
    test_data  = torch.load(f'{RESULTS_DIR}/classifier_fold{fold_idx}_test.pt',  weights_only=False)
    
    # Calcular pos_weight = (num_deadlocks) / (num_solvables) en TRAIN
    labels = [d['label'] for d in train_data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    train_loader = DataLoader(FoldDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(FoldDataset(test_data),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Fold {fold_idx}: Train={len(train_data)} | Test={len(test_data)}")
    print(f"Distribución (Train): Solubles={n_pos} | Deadlocks={n_neg}")
    print(f"Pos Weight sugerido: {pos_weight:.2f}")
    
    return train_loader, test_loader, pos_weight
"""))

# ── SECCIÓN 3: Entrenamiento ──────────────────────────────────────────────
cells.append(md("## 2. Bucle de Entrenamiento"))
cells.append(cell("""\
def train_fold(fold_idx, epochs=30, patience=8, lr=1e-3, weight_decay=1e-5):
    print(f"\\n{'='*55}\\n FOLD {fold_idx}/5\\n{'='*55}")
    train_loader, test_loader, pos_w = get_fold_loaders(fold_idx)
    
    model = SokobanResNetClassifier(dropout_p=0.4).to(device)
    
    # Aquí multiplicamos el pos_w por un factor si queremos castigar aún más los Falsos Positivos
    # (Un falso positivo = predecir que es Soluble cuando en realidad es Deadlock).
    # Como priorizamos el RECALL de deadlocks (clase 0), podemos bajar pos_weight, o subirlo si queremos
    # maximizar RECALL de solubles (clase 1). 
    criterion = ClassifierLoss(pos_weight_val=pos_w)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    
    history = {'loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    
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
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        val_loss /= len(test_loader)
        
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.0
            
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(f1)
        history['val_auc'].append(auc)
        
        scheduler.step(f1)
        
        tag = ""
        if f1 > best_f1:
            best_f1 = f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = " ★"
        else:
            patience_ctr += 1
            
        print(f"Ep {epoch:03d} | L: {train_loss:.3f} | vL: {val_loss:.3f} | "
              f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}{tag}")
              
        if patience_ctr >= patience:
            print(f"🛑 Early Stopping en época {epoch}.")
            break
            
    # Guardar modelo
    out = f"{RESULTS_DIR}/classifier_fold{fold_idx}_model.pt"
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), out)
    print(f"✅ Mejor modelo guardado en {out} con F1={best_f1:.3f}")
    
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
