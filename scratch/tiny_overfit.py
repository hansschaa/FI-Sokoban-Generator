import sys
import os
import glob
sys.path.append('surrogate_models')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.resnet import SokobanSEResNetClassifier, ClassifierLoss
from data.prepare_classifier import parse_sok_file, SOLVABLES_DIR, UNSOLVABLES_FILE
from data.board_utils import encode_board

RESULTS_DIR = "surrogate_models/results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoldDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["tensor"].float(), torch.tensor(item["is_solvable"], dtype=torch.float32)

def generate_tiny_dataset():
    print("Generando 20 ejemplos al vuelo desde los archivos .sok...")
    solvable_files = glob.glob(os.path.join(SOLVABLES_DIR, "**", "*.sok"), recursive=True)
    solvables = parse_sok_file(solvable_files[0], default_label=1)[:10]
    deadlocks = parse_sok_file(UNSOLVABLES_FILE, default_label=0)[:10]
    
    tiny_data = []
    for item in solvables + deadlocks:
        tensor = encode_board(item["board_str"])
        tiny_data.append({
            "tensor": torch.from_numpy(tensor),
            "is_solvable": item["label"]
        })
    return tiny_data

def main():
    print("Prueba de sobreajuste (Tiny Dataset)...")
    try:
        train_data = torch.load(f"{RESULTS_DIR}/classifier_fold1_train.pt", weights_only=False)
        pos_examples = [d for d in train_data if d["is_solvable"] == 1][:10]
        neg_examples = [d for d in train_data if d["is_solvable"] == 0][:10]
        tiny_data = pos_examples + neg_examples
    except FileNotFoundError:
        print("Dataset .pt no encontrado, generándolo en memoria...")
        tiny_data = generate_tiny_dataset()

    if len(tiny_data) < 20:
        print("No hay suficientes datos para armar el tiny dataset.")
        return

    tiny_loader = DataLoader(FoldDataset(tiny_data), batch_size=20, shuffle=True)

    # Inicializar modelo SIN dropout
    model = SokobanSEResNetClassifier(dropout_p=0.0).to(device)
    criterion = ClassifierLoss(pos_weight_val=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("Entrenando por 200 épocas en 20 ejemplos...")
    model.train()
    
    for epoch in range(1, 201):
        for tensors, labels in tiny_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean().item()
            
        if epoch % 20 == 0:
            print(f"Época {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

    # Verificación final
    model.eval()
    with torch.no_grad():
        for tensors, labels in tiny_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            logits = model(tensors)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            final_acc = (preds == labels).float().mean().item()
            print(f"\nExactitud final en los 20 ejemplos: {final_acc*100:.1f}%")
            if final_acc == 1.0:
                print("✅ El modelo logró memorizar los 20 ejemplos perfectamente. (No es problema de capacidad)")
            else:
                print("❌ El modelo falló en memorizar 20 ejemplos. (Posible bug en arquitectura/optimización/datos)")

if __name__ == "__main__":
    main()
