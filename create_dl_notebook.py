import json
import os

def code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

def md_cell(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

cells.append(md_cell("# Entrenamiento de Surrogate Models para Sokoban (Deep Learning)\nEn este notebook entrenaremos y compararemos 3 arquitecturas diferentes para predecir el número de `pushes` (empujes) basándonos en tensores espaciales 2D del tablero."))

cells.append(code_cell("""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Forzar el uso de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=====================================")
print(f" DISPOSITIVO DE ENTRENAMIENTO: {device.type.upper()}")
print(f"=====================================")

if device.type != 'cuda':
    print("⚠️ ADVERTENCIA: No se detectó una GPU CUDA compatible. El entrenamiento será muy lento.")
else:
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
"""))

cells.append(md_cell("## 1. Carga de Datos y DataLoaders\nAquí preparamos dos estrategias de batching: \n- **Padding Centrado** (Para CNN normal y ResNet)\n- **Batching Dinámico por Dimensiones** sin padding (Para la FCN)"))

cells.append(code_cell("""\
# ---------------------------------------------------------
# ESTRATEGIA 1: Padding Centrado (Para CNN y ResNet)
# ---------------------------------------------------------
class PaddedSokobanDataset(Dataset):
    def __init__(self, data_list, max_h=25, max_w=25):
        self.data = data_list
        self.max_h = max_h
        self.max_w = max_w
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        t = item['tensor']
        pushes = float(item['pushes'])
        
        _, h, w = t.shape
        pad_top = (self.max_h - h) // 2
        pad_bottom = self.max_h - h - pad_top
        pad_left = (self.max_w - w) // 2
        pad_right = self.max_w - w - pad_left
        
        t_padded = torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        
        # Rellenar con MUROS (Canal 0) en las zonas de padding
        if pad_top > 0: t_padded[0, :pad_top, :] = 1.0
        if pad_bottom > 0: t_padded[0, -pad_bottom:, :] = 1.0
        if pad_left > 0: t_padded[0, :, :pad_left] = 1.0
        if pad_right > 0: t_padded[0, :, -pad_right:] = 1.0
        
        return t_padded, torch.tensor(pushes, dtype=torch.float32)

# ---------------------------------------------------------
# ESTRATEGIA 2: FCN Variable Batching (Sin Padding)
# ---------------------------------------------------------
def collate_fcn(batch):
    tensors = torch.stack([b[0] for b in batch])
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return tensors, targets

class FCNSokobanDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]['tensor'], float(self.data[idx]['pushes'])

class ShapeBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_list, batch_size):
        self.shape_indices = defaultdict(list)
        for idx, item in enumerate(data_list):
            self.shape_indices[item['tensor'].shape].append(idx)
        self.batch_size = batch_size
            
    def __iter__(self):
        batches = []
        for shape, indices in self.shape_indices.items():
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                if len(batch_indices) > 1: # Prevenir error de BatchNorm con batch_size=1
                    batches.append(batch_indices)
        np.random.shuffle(batches)
        for b in batches: yield b
            
    def __len__(self):
        return sum(len(indices) // self.batch_size for indices in self.shape_indices.values())

def get_fold_dataloaders(fold_idx):
    train_data = torch.load(f'dl_dataset_fold{fold_idx}_train.pt')
    test_data = torch.load(f'dl_dataset_fold{fold_idx}_test.pt')
    
    train_loader_padded = DataLoader(PaddedSokobanDataset(train_data), batch_size=128, shuffle=True)
    test_loader_padded = DataLoader(PaddedSokobanDataset(test_data), batch_size=128, shuffle=False)
    
    train_sampler_fcn = ShapeBatchSampler(train_data, batch_size=64)
    train_loader_fcn = DataLoader(FCNSokobanDataset(train_data), batch_sampler=train_sampler_fcn, collate_fn=collate_fcn)
    
    test_sampler_fcn = ShapeBatchSampler(test_data, batch_size=64)
    test_loader_fcn = DataLoader(FCNSokobanDataset(test_data), batch_sampler=test_sampler_fcn, collate_fn=collate_fcn)
    
    return train_loader_padded, test_loader_padded, train_loader_fcn, test_loader_fcn

"""))

cells.append(md_cell("## 2. Definición de Arquitecturas\nLas tres redes a comparar experimentalmente."))

cells.append(code_cell("""\
# 1. CNN Clásica
class NormalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze()

# 2. FCN con entrada variable
class FCNVariable(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x).squeeze()

# 3. ResNet Simple
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = BasicBlock(32, 64)
        self.layer2 = BasicBlock(64, 64)
        self.layer3 = BasicBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x).squeeze()
"""))

cells.append(md_cell("## 3. Función de Entrenamiento\nUtilizamos **Huber Loss** (Smooth L1) para estabilizar el entrenamiento frente a tableros atípicamente difíciles (>100 pushes), y reportamos el MAE (Mean Absolute Error) en pushes para que sea interpretable humanamente."))

cells.append(code_cell("""\
def train_model(model_class, train_loader, test_loader, epochs=150, name="Model", patience=15):
    import copy
    model = model_class().to(device)
    criterion = nn.HuberLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_history, test_history = [], []
    best_mae = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    print(f"\\n--- Iniciando Entrenamiento: {name} ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        total_mae = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                mae = torch.abs(outputs - targets).mean().item()
                total_mae += mae * inputs.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_mae = total_mae / len(test_loader.dataset)
        train_history.append(avg_loss)
        test_history.append(avg_mae)
        
        scheduler.step(avg_mae)
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = "🌟 Mejoró"
        else:
            patience_counter += 1
            improved = f"⚠️ Estancado ({patience_counter}/{patience})"
            
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_loss:.2f} | Test MAE: {avg_mae:.2f} empujes | LR: {optimizer.param_groups[0]['lr']:.2e} | {improved}")
        
        if patience_counter >= patience:
            print(f"🛑 Early Stopping activado en la época {epoch+1}. Restaurando mejores pesos (MAE: {best_mae:.2f}).")
            model.load_state_dict(best_weights)
            break
            
    print(f"Tiempo del fold: {(time.time() - start_time)/60:.1f} minutos")
    return test_history

def run_cv(model_class, model_name, is_fcn=False):
    print(f"\\n{'='*50}\\n INICIANDO CROSS-VALIDATION: {model_name}\\n{'='*50}")
    all_histories = []
    
    for fold in range(1, 6):
        train_pad, test_pad, train_fcn, test_fcn = get_fold_dataloaders(fold)
        
        if is_fcn:
            train_l, test_l = train_fcn, test_fcn
        else:
            train_l, test_l = train_pad, test_pad
            
        history = train_model(model_class, train_l, test_l, epochs=EPOCHS, patience=15, name=f"{model_name} (Fold {fold})")
        all_histories.append(history)
        
    return all_histories
"""))

cells.append(md_cell("## 4. Ejecutar y Comparar (GPU)"))

cells.append(code_cell("""\
EPOCHS = 150

print("Entrenando CNN Clásica (5-Fold CV)...")
cnn_cv = run_cv(NormalCNN, "CNN Clásica (Padding)")

print("\\nEntrenando FCN Variable (5-Fold CV)...")
fcn_cv = run_cv(FCNVariable, "FCN Variable (Sin Padding)", is_fcn=True)

print("\\nEntrenando ResNet Simple (5-Fold CV)...")
resnet_cv = run_cv(SimpleResNet, "ResNet Simple (Padding)")
"""))

cells.append(code_cell("""\
def pad_histories(histories, max_len=None):
    if max_len is None:
        max_len = max(len(h) for h in histories)
    padded = []
    for h in histories:
        if len(h) < max_len:
            padded.append(h + [h[-1]] * (max_len - len(h)))
        else:
            padded.append(h[:max_len])
    return np.array(padded)

cnn_arr = pad_histories(cnn_cv)
fcn_arr = pad_histories(fcn_cv)
resnet_arr = pad_histories(resnet_cv)

cnn_mean, cnn_std = cnn_arr.mean(axis=0), cnn_arr.std(axis=0)
fcn_mean, fcn_std = fcn_arr.mean(axis=0), fcn_arr.std(axis=0)
resnet_mean, resnet_std = resnet_arr.mean(axis=0), resnet_arr.std(axis=0)

# Gráfico comparativo de Test MAE con Cross-Validation
plt.figure(figsize=(12, 7))
epochs_range = np.arange(1, len(cnn_mean) + 1)
plt.plot(epochs_range, cnn_mean, label="CNN Clásica", color='blue')
plt.fill_between(epochs_range, cnn_mean - cnn_std, cnn_mean + cnn_std, color='blue', alpha=0.2)

epochs_range_fcn = np.arange(1, len(fcn_mean) + 1)
plt.plot(epochs_range_fcn, fcn_mean, label="FCN Variable", color='orange')
plt.fill_between(epochs_range_fcn, fcn_mean - fcn_std, fcn_mean + fcn_std, color='orange', alpha=0.2)

epochs_range_res = np.arange(1, len(resnet_mean) + 1)
plt.plot(epochs_range_res, resnet_mean, label="ResNet Simple", color='green')
plt.fill_between(epochs_range_res, resnet_mean - resnet_std, resnet_mean + resnet_std, color='green', alpha=0.2)

plt.title("Comparación 5-Fold CV (Test MAE Medio ± 1 Std)")
plt.xlabel("Épocas")
plt.ylabel("Test MAE (Margen de error en empujes)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"\\nResultados Finales CV (Mejor MAE Promedio):")
print(f"CNN Clásica: {np.min(cnn_mean):.2f} ± {cnn_std[np.argmin(cnn_mean)]:.2f}")
print(f"FCN Variable: {np.min(fcn_mean):.2f} ± {fcn_std[np.argmin(fcn_mean)]:.2f}")
print(f"ResNet Simple: {np.min(resnet_mean):.2f} ± {resnet_std[np.argmin(resnet_mean)]:.2f}")
"""))

cells.append(md_cell("## 5. Optimización de Hiperparámetros (Optuna)\nAplicaremos Optuna sobre nuestra mejor arquitectura (ResNet Simple) para encontrar la tasa de aprendizaje (LR) y Weight Decay óptimos."))

cells.append(code_cell("""\
import optuna
import copy

def objective(trial):
    # Optuna usa solo el FOLD 1 para buscar hiperparámetros rápido
    train_pad, test_pad, _, _ = get_fold_dataloaders(1)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    model = SimpleResNet().to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epochs = 40 # Épocas reducidas para búsqueda rápida
    patience = 7
    best_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_pad:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        model.eval()
        total_mae = 0
        with torch.no_grad():
            for inputs, targets in test_pad:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                mae = torch.abs(outputs - targets).mean().item()
                total_mae += mae * inputs.size(0)
                
        avg_mae = total_mae / len(test_pad.dataset)
        scheduler.step(avg_mae)
        
        # Reportar a Optuna para Pruning
        trial.report(avg_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if avg_mae < best_mae:
            best_mae = avg_mae
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_mae

print("Iniciando estudio de Optuna (15 trials). ¡Esto tomará un tiempo!...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("\\\\n✅ Mejores hiperparámetros encontrados:")
print(study.best_params)
print(f"Mejor MAE alcanzado: {study.best_value:.2f} empujes")
"""))

cells.append(code_cell("""\
# Gráfico de optimización de Optuna
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Historia de Optimización (Optuna)")
plt.show()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.title("Importancia de los Hiperparámetros")
plt.show()
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("/home/hanss/FI-sokoban-generator/train_surrogate_models.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook de entrenamiento generado correctamente.")
