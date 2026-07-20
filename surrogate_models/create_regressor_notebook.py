"""
create_regressor_notebook.py
----------------------------
Genera el Jupyter Notebook de entrenamiento del Regresor Multi-Head.
Ejecutar desde la raíz del proyecto:
    venv/bin/python surrogate_models/create_regressor_notebook.py
"""
import json, os

def cell(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── SECCIÓN 0: Cabecera ────────────────────────────────────────────────────
cells.append(md("""# Surrogate Regressor: ResNet Multi-Head (5-Fold GroupKFold)
Predice **Pushes** y **Branching Factor** a partir del tensor 5-canales del tablero.

**Anti-leakage garantizado**: cada fold agrupa por `shell_hash`, nunca un cascarón aparece en train y test simultáneamente.
"""))

# ── SECCIÓN 1: Setup ───────────────────────────────────────────────────────
cells.append(cell("""\
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # para importar models/resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device.type.upper()}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

# ── SECCIÓN 2: Dataset ─────────────────────────────────────────────────────
cells.append(md("## 1. Dataset y DataLoaders"))
cells.append(cell("""\
from models.resnet import SokobanResNetRegressor, MultiHeadRegressorLoss

RESULTS_DIR = '../results'

class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['tensor'].float(),
            torch.tensor(item['pushes_norm'],    dtype=torch.float32),
            torch.tensor(item['branch_norm'],    dtype=torch.float32),
            torch.tensor(item['pushes_raw'],     dtype=torch.float32),
            torch.tensor(item['branch_raw'],     dtype=torch.float32),
        )

def get_fold_loaders(fold_idx, batch_size=128):
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler

    train_data = torch.load(f'{RESULTS_DIR}/regressor_fold{fold_idx}_train.pt', weights_only=False)
    test_data  = torch.load(f'{RESULTS_DIR}/regressor_fold{fold_idx}_test.pt',  weights_only=False)
    stats      = torch.load(f'{RESULTS_DIR}/regressor_fold{fold_idx}_stats.pt', weights_only=False)

    # WeightedRandomSampler: cada bucket tiene la misma probabilidad de aparecer
    # en cada batch, sin importar cuántos tableros tenga.
    # Con ratio ~6:1 esto suaviza el sesgo hacia tableros fáciles.
    bucket_counts = Counter(d['bucket'] for d in train_data)
    sample_weights = [1.0 / bucket_counts[d['bucket']] for d in train_data]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Nota: sampler ya implica shuffling, NO usar shuffle=True simultáneamente
    train_loader = DataLoader(FoldDataset(train_data), batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(FoldDataset(test_data),  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Fold {fold_idx} — Train: {len(train_data):,} | Test: {len(test_data):,}")
    print(f"  Distribución de buckets (train, sin aug factor):")
    raw_counts = Counter(d['bucket'] for d in train_data)
    for k in sorted(raw_counts): print(f"    {k}: {raw_counts[k]:,}")
    return train_loader, test_loader, stats

print("✅ Funciones de carga listas.")
"""))


# ── SECCIÓN 3: Función de Entrenamiento ────────────────────────────────────
cells.append(md("## 2. Función de Entrenamiento (con Early Stopping y Scheduler)"))
cells.append(cell("""\
def train_fold(fold_idx, epochs=150, patience=15, lr=1e-3, weight_decay=1e-5):
    print(f"\\n{'='*55}")
    print(f" FOLD {fold_idx}/5")
    print(f"{'='*55}")

    train_loader, test_loader, stats = get_fold_loaders(fold_idx)
    p_mean, p_std = stats['pushes_mean'], stats['pushes_std']
    b_mean, b_std = stats['branch_mean'], stats['branch_std']

    model     = SokobanResNetRegressor(dropout_p=0.4).to(device)
    criterion = MultiHeadRegressorLoss(w_pushes=1.0, w_branch=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_mae_pushes = float('inf')
    best_weights    = copy.deepcopy(model.state_dict())
    patience_ctr    = 0
    history         = []
    t0              = time.time()

    for epoch in range(1, epochs + 1):
        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for tensors, p_norm, b_norm, _, _ in train_loader:
            tensors = tensors.to(device)
            p_norm  = p_norm.to(device)
            b_norm  = b_norm.to(device)
            optimizer.zero_grad()
            p_pred, b_pred = model(tensors)
            loss, _ = criterion(p_pred, p_norm, b_pred, b_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        # ── EVAL ───────────────────────────────────────────────────────────
        model.eval()
        total_mae_p, total_mae_b, n = 0.0, 0.0, 0
        with torch.no_grad():
            for tensors, _, _, p_raw, b_raw in test_loader:
                tensors = tensors.to(device)
                p_pred, b_pred = model(tensors)
                # Desnormalizar para calcular MAE en espacio original
                p_desnorm = p_pred.cpu() * p_std + p_mean
                b_desnorm = b_pred.cpu() * b_std + b_mean
                total_mae_p += torch.abs(p_desnorm - p_raw).sum().item()
                total_mae_b += torch.abs(b_desnorm - b_raw).sum().item()
                n += len(p_raw)

        mae_p = total_mae_p / n
        mae_b = total_mae_b / n
        avg_loss = epoch_loss / len(train_loader)
        history.append({'epoch': epoch, 'loss': avg_loss, 'mae_pushes': mae_p, 'mae_branch': mae_b})
        scheduler.step(mae_p)

        if mae_p < best_mae_pushes:
            best_mae_pushes = mae_p
            best_weights    = copy.deepcopy(model.state_dict())
            patience_ctr    = 0
            tag = "🌟"
        else:
            patience_ctr += 1
            tag = f"⚠️ ({patience_ctr}/{patience})"

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.3f} | MAE Pushes: {mae_p:.2f} | MAE Branch: {mae_b:.3f} | LR: {lr_now:.1e} | {tag}")

        if patience_ctr >= patience:
            print(f"🛑 Early Stopping en época {epoch}. Mejor MAE Pushes: {best_mae_pushes:.2f}")
            break

    model.load_state_dict(best_weights)
    print(f"⏱  Tiempo: {(time.time()-t0)/60:.1f} min | Mejor MAE Pushes: {best_mae_pushes:.2f} empujes")
    return model, history, best_mae_pushes

print("✅ Función de entrenamiento lista.")
"""))

# ── SECCIÓN 4: Ejecutar CV ─────────────────────────────────────────────────
cells.append(md("## 3. Ejecutar 5-Fold Cross-Validation"))
cells.append(cell("""\
EPOCHS = 150

all_histories    = []
all_best_maes    = []
all_models       = []

for fold in range(1, 6):
    model, history, best_mae = train_fold(fold, epochs=EPOCHS, patience=15)
    all_histories.append(history)
    all_best_maes.append(best_mae)
    all_models.append(model)

print("\\n" + "="*55)
print(" RESULTADOS FINALES 5-FOLD CV")
print("="*55)
for i, mae in enumerate(all_best_maes, 1):
    print(f"  Fold {i}: MAE Pushes = {mae:.2f} empujes")
mean_mae = np.mean(all_best_maes)
std_mae  = np.std(all_best_maes)
print(f"\\n  Promedio: {mean_mae:.2f} ± {std_mae:.2f} empujes")
print(f"\\n✅ Cross-Validation completado.")
"""))

# ── SECCIÓN 5: Gráficos ────────────────────────────────────────────────────
cells.append(md("## 4. Gráficos de Convergencia"))
cells.append(cell("""\
def pad_metric(histories, key):
    max_len = max(len(h) for h in histories)
    arr = []
    for h in histories:
        vals = [e[key] for e in h]
        if len(vals) < max_len:
            vals += [vals[-1]] * (max_len - len(vals))
        arr.append(vals[:max_len])
    return np.array(arr)

mae_arr   = pad_metric(all_histories, 'mae_pushes')
branch_arr = pad_metric(all_histories, 'mae_branch')
loss_arr  = pad_metric(all_histories, 'loss')

mean_mae_hist = mae_arr.mean(axis=0)
std_mae_hist  = mae_arr.std(axis=0)
x = np.arange(1, len(mean_mae_hist)+1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot MAE Pushes
ax = axes[0]
ax.plot(x, mean_mae_hist, color='steelblue', linewidth=2, label='MAE Pushes (media)')
ax.fill_between(x, mean_mae_hist - std_mae_hist, mean_mae_hist + std_mae_hist, alpha=0.25, color='steelblue')
ax.axhline(mean_mae_hist.min(), linestyle='--', color='red', alpha=0.7, label=f'Mejor: {mean_mae_hist.min():.2f}')
ax.set_title('MAE Pushes (5-Fold CV)')
ax.set_xlabel('Épocas')
ax.set_ylabel('MAE (empujes reales)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot Training Loss
ax = axes[1]
mean_loss = loss_arr.mean(axis=0)
std_loss  = loss_arr.std(axis=0)
ax.plot(x, mean_loss, color='darkorange', linewidth=2, label='Loss (media)')
ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, alpha=0.25, color='darkorange')
ax.set_title('Training Loss (5-Fold CV)')
ax.set_xlabel('Épocas')
ax.set_ylabel('Asymmetric Huber Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'ResNet Multi-Head Regressor — MAE final: {mean_mae:.2f} ± {std_mae:.2f} empujes', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/regressor_cv_curves.png', dpi=150)
plt.show()
print("✅ Gráfico guardado en results/regressor_cv_curves.png")
"""))

# ── SECCIÓN 6: Guardar el mejor modelo ────────────────────────────────────
cells.append(md("## 5. Guardar el Mejor Modelo"))
cells.append(cell("""\
best_fold_idx = int(np.argmin(all_best_maes))
best_model    = all_models[best_fold_idx]
save_path     = f'../results/best_regressor_fold{best_fold_idx+1}.pt'

torch.save(best_model.state_dict(), save_path)
print(f"✅ Mejor modelo (Fold {best_fold_idx+1}, MAE={all_best_maes[best_fold_idx]:.2f}) guardado en:")
print(f"   {save_path}")
"""))

# ── Ensamblar notebook ─────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = os.path.join(os.path.dirname(__file__), "notebooks", "train_regressor.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✅ Notebook generado: {out}")
