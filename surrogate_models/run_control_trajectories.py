"""
run_control_trajectories.py
---------------------------
Experimento de control de trayectorias sin poda (Pruning Audit) para verificar empíricamente
el sesgo del MedianPruner en las etapas tempranas del entrenamiento del Clasificador Contrastivo.

Ejecuta 5 configuraciones diversas (ganadoras de v2 y alternativas de inicio lento/alta regularización)
durante 15 épocas completas sin podar. Genera un análisis post-mortem comparando:
 1. Ranking y mediana en Época 3 y Época 6 (dónde habría podado el algoritmo anterior).
 2. Ranking final y F_0.5 óptimo al finalizar la Época 15.
 3. Identificación precisa de falsos negativos (modelos que habrían sido asesinados prematurmente y superan el baseline).

Ejecución (en una de las PCs del clúster):
    venv/bin/python surrogate_models/run_control_trajectories.py
"""

import sys, os, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

from models.resnet import SokobanSEResNetClassifier

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FOLD        = 1
MAX_EPOCHS  = 15
BETA        = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*70}")
print("  AUDITORÍA DE PRUNING — EXPERIMENTO DE CONTROL DE TRAYECTORIAS (5 CONF)")
print(f"  Dispositivo: {device.type.upper()} | Épocas por modelo: {MAX_EPOCHS} (Sin Poda)")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET EN MEMORIA CON D4 AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveMemoryDataset(Dataset):
    def __init__(self, X_tensor, y_tensor, t_tensor, is_train=False):
        self.X = X_tensor
        self.y = y_tensor
        self.t = t_tensor
        self.is_train = is_train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            k = random.randint(0, 3)
            flip = random.choice([True, False])
            x = torch.rot90(x, k, [1, 2])
            if flip:
                x = torch.flip(x, [2])
        return x, self.y[idx], self.t[idx]

print("Cargando dataset contrastivo Fold 1 en memoria...")
_train_X = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_X_train.pt"), map_location='cpu')
_train_y = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_y_train.pt"), map_location='cpu')
_train_t = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_t_train.pt"), map_location='cpu')

_val_X = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_X_test.pt"), map_location='cpu')
_val_y = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_y_test.pt"), map_location='cpu')
_val_t = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_t_test.pt"), map_location='cpu')

num_pos = (_train_y == 1).sum().item()
num_neg = (_train_y == 0).sum().item()
_pos_weight_val = num_neg / max(1, num_pos)

_train_dataset = ContrastiveMemoryDataset(_train_X, _train_y, _train_t, is_train=True)
_val_dataset   = ContrastiveMemoryDataset(_val_X, _val_y, _val_t, is_train=False)

def make_loaders(batch_size):
    return (
        DataLoader(_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(_val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    )

# ─────────────────────────────────────────────────────────────────────────────
# LAS 5 CONFIGURACIONES DE AUDITORÍA
# ─────────────────────────────────────────────────────────────────────────────
configs = [
    {"id": "Config_1 (Top v2 #58)", "lr": 5.225e-4, "weight_decay": 3.20e-6, "dropout_p": 0.13, "batch_size": 64},
    {"id": "Config_2 (Start v2 #2)", "lr": 6.240e-4, "weight_decay": 2.19e-4, "dropout_p": 0.38, "batch_size": 128},
    {"id": "Config_3 (High Drop/LR)", "lr": 1.500e-3, "weight_decay": 1.00e-5, "dropout_p": 0.48, "batch_size": 128},
    {"id": "Config_4 (Slow Start)",    "lr": 8.000e-5, "weight_decay": 5.00e-5, "dropout_p": 0.20, "batch_size": 64},
    {"id": "Config_5 (High Decay)",    "lr": 2.000e-3, "weight_decay": 5.00e-3, "dropout_p": 0.25, "batch_size": 256}
]

trajectories = {}
best_metrics_summary = []

for c in configs:
    cid = c["id"]
    print(f"\n🚀 Entrenando {cid} -> lr={c['lr']:.2e} | wd={c['weight_decay']:.2e} | drop={c['dropout_p']} | bs={c['batch_size']}")
    
    set_seed(42 + len(trajectories)) # Diferente semilla de inicialización por config pero reproducible
    train_loader, val_loader = make_loaders(c["batch_size"])
    
    model = SokobanSEResNetClassifier(dropout_p=c["dropout_p"], in_channels=12).to(device)
    pos_weight_tensor = torch.tensor([_pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    history_f05 = []
    best_f05_val = 0.0
    best_info = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        scheduler.step()
        
        # Validación y barrido de umbral
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                probs   = torch.sigmoid(model(X_batch)).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(y_batch.numpy())
                
        all_probs   = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        ep_best_f = 0.0
        ep_best_th = 0.5
        ep_prec = 0.0
        ep_rec  = 0.0
        for th in np.arange(0.50, 0.96, 0.05):
            preds = (all_probs >= th).astype(float)
            fb = fbeta_score(all_targets, preds, beta=BETA, zero_division=0)
            if fb > ep_best_f:
                ep_best_f  = fb
                ep_best_th = th
                ep_prec    = precision_score(all_targets, preds, zero_division=0)
                ep_rec     = recall_score(all_targets, preds, zero_division=0)
                
        history_f05.append(ep_best_f)
        print(f"   Época {epoch:02d}/{MAX_EPOCHS} | F0.5={ep_best_f:.4f} (umbral={ep_best_th:.2f}, Prec={ep_prec:.4f}, Rec={ep_rec:.4f})")
        
        if ep_best_f > best_f05_val:
            best_f05_val = ep_best_f
            best_info = {"epoch": epoch, "f05": ep_best_f, "threshold": ep_best_th, "precision": ep_prec, "recall": ep_rec}
            
    trajectories[cid] = history_f05
    best_metrics_summary.append({"id": cid, **best_info, **c})

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE CORRELACIÓN Y PODA TEMPRANA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("  AUDITORÍA DE PODA: TRAYECTORIAS Y DECISIONES HIPOTÉTICAS DEL PRUNER")
print("="*80)

df_traj = pd.DataFrame(trajectories)
df_traj.index = [f"Epoca_{i+1:02d}" for i in range(MAX_EPOCHS)]

# Mediana en época 3 y época 6
med_ep3 = df_traj.loc["Epoca_03"].median()
med_ep6 = df_traj.loc["Epoca_06"].median()

print("\n📈 Trayectorias de F_0.5 por Modelo:")
print(df_traj.to_string(float_format=lambda x: f"{x:.4f}"))
print("-" * 80)

print(f"\n🔍 Análisis en Época 3 (Mediana del grupo: {med_ep3:.4f}):")
for cid in trajectories.keys():
    score_3 = df_traj.loc["Epoca_03", cid]
    estado_3 = "💀 PODADO (Matado en v2)" if score_3 < med_ep3 else "✅ SOBREVIVE"
    print(f"   * {cid:<24} | F0.5 @ Ep3: {score_3:.4f} -> {estado_3}")

print(f"\n🔍 Análisis en Época 6 (Mediana del grupo: {med_ep6:.4f}):")
for cid in trajectories.keys():
    score_6 = df_traj.loc["Epoca_06", cid]
    estado_6 = "💀 PODADO" if score_6 < med_ep6 else "✅ SOBREVIVE"
    print(f"   * {cid:<24} | F0.5 @ Ep6: {score_6:.4f} -> {estado_6}")

print("\n" + "="*80)
print("  RANKING DEFINITIVO EN ÉPOCA 15 VS ÉPOCAS TEMPRANAS")
print("="*80)
sorted_final = sorted(best_metrics_summary, key=lambda x: x["f05"], reverse=True)
for rank, item in enumerate(sorted_final, 1):
    cid = item["id"]
    s_3 = df_traj.loc["Epoca_03", cid]
    s_6 = df_traj.loc["Epoca_06", cid]
    flag_murder = "⚠️ FALSO NEGATIVO (Habría sido matado en Ep 3!)" if s_3 < med_ep3 and rank <= 2 else ""
    print(f" #{rank} | {cid:<24} | F0.5 Final: {item['f05']:.4f} (Ep {item['epoch']:02d}) | Ep3: {s_3:.4f} | Ep6: {s_6:.4f} {flag_murder}")

out_csv = os.path.join(RESULTS_DIR, "control_trajectories_audit.csv")
df_traj.to_csv(out_csv)
out_json = os.path.join(RESULTS_DIR, "control_trajectories_summary.json")
with open(out_json, "w") as f:
    json.dump({"summary": best_metrics_summary, "trajectories": trajectories}, f, indent=2)
print(f"\n✅ Datos de auditoría guardados en {RESULTS_DIR}")
print("========================================================================\n")
