import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.resnet import SokobanResNetClassifier, ClassifierLoss

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

class FoldDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["tensor"], item["is_solvable"]

def main():
    print("=== EXAMEN PILOTO: CLASIFICADOR ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device.type.upper()}")
    
    train_path = os.path.join(RESULTS_DIR, "classifier_fold1_train.pt")
    test_path  = os.path.join(RESULTS_DIR, "classifier_fold1_test.pt")
    
    if not os.path.exists(train_path):
        print(f"❌ Error: No se encontró {train_path}")
        print("Debes ejecutar primero: python3 surrogate_models/data/prepare_classifier.py")
        sys.exit(1)
        
    print("Cargando datos...")
    train_data = torch.load(train_path, weights_only=False)
    test_data  = torch.load(test_path, weights_only=False)
    
    train_loader = DataLoader(FoldDataset(train_data), batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(FoldDataset(test_data), batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    
    # Calcular class weights para ClassifierLoss
    targets = [d["is_solvable"] for d in train_data]
    N = len(targets)
    N_pos = sum(targets)
    N_neg = N - N_pos
    pos_weight = N_neg / N_pos if N_pos > 0 else 1.0
    print(f"Pesos de clase: pos_weight={pos_weight:.2f} (para contrarrestar el exceso de deadlocks)")

    model = SokobanResNetClassifier().to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print(f"\nIniciando entrenamiento piloto de {epochs} épocas...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for tensors, labels in train_loader:
            tensors = tensors.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            
        train_loss /= len(train_data)
        
        # Test
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for tensors, labels in test_loader:
                tensors = tensors.to(device)
                labels = labels.to(device).float()
                
                logits = model(tensors)
                loss = criterion(logits, labels)
                test_loss += loss.item() * len(labels)
                
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        test_loss /= len(test_data)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        elapsed = time.time() - start_time
        print(f"Ep {epoch:02d} | T: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")

    print("\n✅ Examen piloto finalizado. Si el Acc y F1 suben, ¡el modelo está aprendiendo a distinguir solubles de insolubles!")

if __name__ == "__main__":
    main()
