"""
ablation_study.py
-----------------
Entrena y compara 3 arquitecturas (CNN, ResNet, SE-ResNet) usando los mismos folds (Cross Validation)
para prevenir data leakage. Mismos hiperparámetros para todas las redes.
"""

import sys, os, json, argparse, time, copy, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Modelos
from models.cnn import SokobanCNNRegressor, SokobanCNNClassifier
from models.resnet import SokobanResNetRegressor, SokobanSEResNetRegressor, SokobanResNetRegressorNoSE
from models.resnet import SokobanResNetClassifier, SokobanSEResNetClassifier, SokobanSEResNetClassifierNoSE
from models.resnet import ClassifierLoss

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN BASELINE (Hiperparámetros Neutros)
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 60
BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

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
            torch.tensor(item['pushes_raw'],  dtype=torch.float32),
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
# RUTINAS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def run_training_loop(model, train_loader, val_loader, criterion, is_classifier, arch_name, fold):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    history = []
    
    print(f"    -> Entrenando {arch_name}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if is_classifier:
                y = batch[1].to(device)
                preds = model(x)
                loss = criterion(preds, y)
            else:
                y_norm = batch[1].to(device)
                w = batch[3].to(device)
                preds = model(x)
                loss_p = criterion(preds, y_norm)
                loss = (loss_p * w).mean()
                
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validación rápida
        model.eval()
        val_metric = 0.0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                if is_classifier:
                    y = batch[1].to(device)
                    preds = model(x)
                    loss = criterion(preds, y)
                    val_metric += loss.item() * x.size(0)
                    total += x.size(0)
                else:
                    y_raw = batch[2].to(device)
                    preds = model(x)
                    # Denormalize pushes: expm1(preds)
                    preds_raw = torch.expm1(preds)
                    mae = torch.abs(preds_raw - y_raw).sum().item()
                    val_metric += mae
                    total += x.size(0)
        
        epoch_metric = val_metric / total
        history.append(epoch_metric)
        if epoch % 10 == 0:
            metric_name = "Loss" if is_classifier else "MAE"
            print(f"       Época {epoch:02d}/{EPOCHS} | Val {metric_name}: {epoch_metric:.4f}")
            
    # Liberar memoria de GPU
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return history


def train_ablation(task, folds_to_run):
    is_classifier = (task == "classifier")
    
    archs = {
        "CNN": SokobanCNNClassifier if is_classifier else SokobanCNNRegressor,
        "ResNet": SokobanSEResNetClassifierNoSE if is_classifier else SokobanResNetRegressorNoSE,
        "SEResNet": SokobanSEResNetClassifier if is_classifier else SokobanSEResNetRegressor
    }
    
    criterion = ClassifierLoss(pos_weight_val=3.0) if is_classifier else nn.HuberLoss(reduction='none')
    
    for fold in folds_to_run:
        print(f"\n[{'─'*40}]")
        print(f"  INICIANDO FOLD {fold}/5 ({task.upper()})")
        print(f"[{'─'*40}]")
        
        # Cargar datos para prevenir leakage
        train_path = os.path.join(RESULTS_DIR, f"{task}_fold{fold}_train.pt")
        val_path   = os.path.join(RESULTS_DIR, f"{task}_fold{fold}_val.pt")
        
        if not os.path.exists(train_path):
            print(f"⚠️ Saltando Fold {fold}: No existe {train_path}")
            continue
            
        print("Cargando datasets del disco...")
        train_data = torch.load(train_path, weights_only=False)
        val_data   = torch.load(val_path, weights_only=False)
        
        train_ds = ClassifierDataset(train_data) if is_classifier else RegressorDataset(train_data)
        val_ds   = ClassifierDataset(val_data)  if is_classifier else RegressorDataset(val_data)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        
        for arch_name, ModelClass in archs.items():
            out_json = os.path.join(RESULTS_DIR, f"ablation_{task}_{arch_name}_fold{fold}.json")
            if os.path.exists(out_json):
                print(f"    -> {arch_name} ya fue entrenado en este fold. Saltando...")
                continue
                
            model = ModelClass(dropout_p=DROPOUT)
            history = run_training_loop(model, train_loader, val_loader, criterion, is_classifier, arch_name, fold)
            
            with open(out_json, "w") as f:
                json.dump({"history": history}, f)
                
        # Limpiar data loaders
        del train_data, val_data, train_ds, val_ds, train_loader, val_loader
        gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# GRAFICAR
# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation(task):
    archs = ["CNN", "ResNet", "SEResNet"]
    colors = {"CNN": "red", "ResNet": "blue", "SEResNet": "green"}
    
    plt.figure(figsize=(10, 6))
    
    for arch in archs:
        all_histories = []
        for fold in range(1, 6):
            fpath = os.path.join(RESULTS_DIR, f"ablation_{task}_{arch}_fold{fold}.json")
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    data = json.load(f)
                    all_histories.append(data["history"])
        
        if not all_histories:
            continue
            
        # Calcular media a través de los folds
        avg_history = np.mean(all_histories, axis=0)
        
        plt.plot(range(1, EPOCHS+1), avg_history, label=f"{arch} (Avg {len(all_histories)} folds)", color=colors[arch], linewidth=2)
    
    plt.title(f"Ablation Study: {task.capitalize()} Architectures")
    plt.xlabel("Epoch")
    ylabel = "Validation BCE Loss" if task == "classifier" else "Validation MAE (Pushes)"
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    out_img = os.path.join(RESULTS_DIR, f"ablation_{task}_plot.png")
    plt.savefig(out_img, dpi=300)
    print(f"\n✅ Gráfico guardado en: {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study para Sokoban Surrogate")
    parser.add_argument("--task", type=str, required=True, choices=["regressor", "classifier"])
    parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="Folds separados por coma")
    parser.add_argument("--plot-only", action="store_true", help="Solo generar gráfico con los jsons existentes")
    args = parser.parse_args()

    folds_to_run = [int(f.strip()) for f in args.folds.split(",")]
    
    if not args.plot_only:
        train_ablation(args.task, folds_to_run)
        
    plot_ablation(args.task)
