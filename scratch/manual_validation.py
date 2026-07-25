import sys
sys.path.append('surrogate_models')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import fbeta_score

from models.resnet import SokobanSEResNetClassifier, ClassifierLoss

RESULTS_DIR = "surrogate_models/results"
FOLD = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoldDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["tensor"].float(), torch.tensor(item["is_solvable"], dtype=torch.float32)

def main():
    print("Prueba de validación a escala real (Fold 1) con hiperparámetros fijos...")
    
    try:
        train_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_train.pt", weights_only=False)
        try:
            val_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_val.pt", weights_only=False)
        except FileNotFoundError:
            val_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_test.pt", weights_only=False)
    except FileNotFoundError as e:
        print(f"Dataset no encontrado: {e}")
        return

    train_loader = DataLoader(FoldDataset(train_data), batch_size=256, shuffle=True, num_workers=4)
    val_loader   = DataLoader(FoldDataset(val_data), batch_size=256, shuffle=False, num_workers=4)

    # Hiperparámetros fijos recomendados
    lr = 5e-4
    pos_weight = 4.3
    epochs = 15

    model = SokobanSEResNetClassifier(dropout_p=0.3).to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Iniciando entrenamiento (lr={lr}, pos_weight={pos_weight}, epochs={epochs})...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_logits = []
        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tensors)
            epoch_logits.extend(logits.detach().cpu().numpy())
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        scheduler.step()
        
        el_arr = np.array(epoch_logits)
        print(f"[Epoch {epoch:2d}] Train Loss: {loss.item():.4f} | Logits: mean={el_arr.mean():.3f} std={el_arr.std():.3f}")

        # Eval validation every 3 epochs and at the end
        if epoch % 3 == 0 or epoch == epochs:
            model.eval()
            all_probs, all_targets = [], []
            with torch.no_grad():
                for tensors, labels in val_loader:
                    tensors = tensors.to(device)
                    logits = model(tensors)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    all_targets.extend(labels.numpy())
            
            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            
            best_f_beta = 0.0
            for thresh in np.arange(0.1, 0.9, 0.05):
                preds = (all_probs >= thresh).astype(float)
                f_beta = fbeta_score(all_targets, preds, beta=0.5, zero_division=0)
                best_f_beta = max(best_f_beta, f_beta)
                
            print(f"  -> Val F0.5 (mejor umbral): {best_f_beta:.4f}")

if __name__ == "__main__":
    main()
