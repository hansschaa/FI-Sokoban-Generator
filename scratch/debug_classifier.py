import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score

import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetClassifier, ClassifierLoss

# =========================================================
# CONFIG
# =========================================================
FOLD = 1
RESULTS_DIR = "surrogate_models/results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reemplaza estos valores con los del mejor trial de Optuna
BEST_PARAMS = {
    'batch_size': 256,
    'dropout_p': 0.246,
    'lr': 0.00042,
    'pos_weight': 1.88,
    'weight_decay': 1.72e-06
}

class FoldDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["tensor"].float(), torch.tensor(item["is_solvable"], dtype=torch.float32)

def main():
    print(f"Cargando dataset (Fold {FOLD})...")
    try:
        train_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_train.pt", weights_only=False)
        try:
            val_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_val.pt", weights_only=False)
        except FileNotFoundError:
            # Fallback for older dataset format on lab machines
            val_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_test.pt", weights_only=False)
    except FileNotFoundError as e:
        print(f"Dataset no encontrado: {e}. Asegúrate de ejecutar este script en la misma ruta donde funciona optuna.")
        return

    train_loader = DataLoader(FoldDataset(train_data), batch_size=BEST_PARAMS['batch_size'], shuffle=True, num_workers=4)
    val_loader   = DataLoader(FoldDataset(val_data), batch_size=BEST_PARAMS['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Train size: {len(train_data)} | Val size: {len(val_data)}")

    model = SokobanSEResNetClassifier(dropout_p=BEST_PARAMS['dropout_p']).to(device)
    criterion = ClassifierLoss(pos_weight_val=BEST_PARAMS['pos_weight'])
    optimizer = optim.AdamW(model.parameters(), lr=BEST_PARAMS['lr'], weight_decay=BEST_PARAMS['weight_decay'])
    
    print("\nEntrenando por 2 épocas (aislado) para diagnosticar...")
    for epoch in range(1, 3):
        model.train()
        for tensors, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(tensors.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            
        print(f"Época {epoch} completada.")

    print("\nEvaluando en validación...")
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for tensors, labels in val_loader:
            logits = model(tensors.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # (a) Histograma de Probabilidades
    plt.figure(figsize=(10, 5))
    plt.hist(all_probs[all_targets == 0], bins=50, alpha=0.5, label='Deadlocks (0)', color='red')
    plt.hist(all_probs[all_targets == 1], bins=50, alpha=0.5, label='Solubles (1)', color='blue')
    plt.axvline(0.5, color='black', linestyle='--', label='Threshold Default (0.5)')
    plt.title('Distribución de Probabilidades Predichas (Validación)')
    plt.xlabel('Probabilidad Predicha (Clase 1)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.savefig("diagnostico_probs.png")
    print("\n✅ Histograma guardado como 'diagnostico_probs.png'")

    # Barrido de umbrales amplio
    best_f_beta = 0.0
    best_thresh = 0.0
    print("\nBuscando el mejor umbral en [0.1, 0.9]...")
    for thresh in np.arange(0.10, 0.95, 0.05):
        preds = (all_probs >= thresh).astype(float)
        f_beta = fbeta_score(all_targets, preds, beta=0.5, zero_division=0)
        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_thresh = thresh
            
    print(f"Mejor umbral: {best_thresh:.2f} (F0.5 = {best_f_beta:.4f})")

    # (b) Matriz de confusión
    preds = (all_probs >= best_thresh).astype(float)
    cm = confusion_matrix(all_targets, preds)
    print("\nMatriz de Confusión (con el mejor umbral):")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    print(f"Precisión: {precision_score(all_targets, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_targets, preds, zero_division=0):.4f}")

    # (c) Verificación manual de algunos errores groseros (falsos positivos)
    print("\nAnalizando algunos errores (Falsos Positivos - Deadlocks predichos como solubles con alta prob):")
    fp_indices = np.where((all_targets == 0) & (all_probs >= best_thresh))[0]
    if len(fp_indices) > 0:
        for idx in fp_indices[:5]:
            print(f"  - FP en idx {idx} | Prob Predicha: {all_probs[idx]:.4f} | Target Real: {all_targets[idx]}")
    else:
        print("  Ningún FP encontrado.")

if __name__ == "__main__":
    main()
