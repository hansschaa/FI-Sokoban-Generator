import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add current dir to sys.path to import resnet
sys.path.append(BASE_DIR)
from models.resnet import SokobanSEResNetClassifier

def generate_histogram():
    print("📊 Generado Histograma Mixto de Separación (Out-of-Fold Test Sets)...")
    
    # Load optimal config
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams_contrastive_classifier.json")
    if os.path.exists(hparams_path):
        with open(hparams_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        params = cfg.get("best_params", cfg.get("params", {}))
        dropout_p = float(params.get("dropout_p", 0.3))
        opt_thresh = float(cfg.get("optimal_threshold", 0.70))
    else:
        dropout_p = 0.4
        opt_thresh = 0.65
        
    probs_solvables = []
    probs_simple = []
    probs_complex = []
    probs_other_deadlocks = []
    
    for fold in range(1, 6):
        model_path = os.path.join(RESULTS_DIR, f"final_contrastive_classifier_fold{fold}.pt")
        x_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_X_test.pt")
        y_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_y_test.pt")
        t_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_t_test.pt")
        
        if not all(os.path.exists(p) for p in [model_path, x_path, y_path, t_path]):
            print(f"⚠️ Saltando Fold {fold} por archivos faltantes.")
            continue
            
        model = SokobanSEResNetClassifier(dropout_p=dropout_p, in_channels=12).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        
        X = torch.load(x_path, map_location='cpu')
        y = torch.load(y_path, map_location='cpu')
        t = torch.load(t_path, map_location='cpu')
        
        dataset = TensorDataset(X, y, t)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        fold_probs, fold_y, fold_t = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_t in loader:
                p = torch.sigmoid(model(batch_x.to(device))).cpu().numpy()
                fold_probs.extend(p)
                fold_y.extend(batch_y.numpy())
                fold_t.extend(batch_t.numpy())
                
        fold_probs = np.array(fold_probs)
        fold_y = np.array(fold_y)
        fold_t = np.array(fold_t)
        
        # Categorize
        solv_mask = (fold_y == 1)
        simple_mask = (fold_y == 0) & (fold_t == 2)
        complex_mask = (fold_y == 0) & (fold_t == 3)
        other_dd_mask = (fold_y == 0) & (fold_t != 2) & (fold_t != 3)
        
        probs_solvables.extend(fold_probs[solv_mask])
        probs_simple.extend(fold_probs[simple_mask])
        probs_complex.extend(fold_probs[complex_mask])
        probs_other_deadlocks.extend(fold_probs[other_dd_mask])
        print(f"   Fold {fold}: {solv_mask.sum()} Solubles | {simple_mask.sum()} Simples | {complex_mask.sum()} Complejos")
        
    # Plotting
    plt.figure(figsize=(11, 6), dpi=300)
    
    bins = np.linspace(0.0, 1.0, 50)
    
    if len(probs_solvables) > 0:
        plt.hist(probs_solvables, bins=bins, alpha=0.6, color='dodgerblue', label=f'Solubles (n={len(probs_solvables)})', density=True, edgecolor='black', linewidth=0.5)
    if len(probs_simple) > 0:
        plt.hist(probs_simple, bins=bins, alpha=0.65, color='orange', label=f'Deadlocks Simples (n={len(probs_simple)})', density=True, edgecolor='black', linewidth=0.5)
    if len(probs_complex) > 0:
        plt.hist(probs_complex, bins=bins, alpha=0.7, color='crimson', label=f'Deadlocks Complejos (n={len(probs_complex)})', density=True, edgecolor='black', linewidth=0.5)
    if len(probs_other_deadlocks) > 0:
        plt.hist(probs_other_deadlocks, bins=bins, alpha=0.4, color='gray', label=f'Deadlocks General (n={len(probs_other_deadlocks)})', density=True, edgecolor='black', linewidth=0.5)
        
    plt.axvline(opt_thresh, color='darkgreen', linestyle='--', linewidth=2.5, label=f'Umbral Calibrado Optuna ({opt_thresh:.2f})')
    plt.axvline(0.50, color='gray', linestyle=':', linewidth=1.5, label='Umbral por defecto (0.50)')
    
    plt.title('Distribución de Probabilidades Predichas por Tipo de Estado (Out-of-Fold CV)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Probabilidad Predicha (Clase 1 = Soluble)', fontsize=12)
    plt.ylabel('Densidad Normalizada', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10.5)
    
    out_path = os.path.join(RESULTS_DIR, "separation_histogram_optuna_v3.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
    print(f"\n✅ ¡Histograma de separación guardado con éxito en: {out_path}!")

if __name__ == "__main__":
    generate_histogram()
