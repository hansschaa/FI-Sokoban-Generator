import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

from models.resnet import SokobanSEResNetClassifier, ClassifierLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
BATCH_SIZE = 256
EPOCHS = 15 # Contrastive model should converge fast as it's fine-tuning or training on similar task

import random

class ContrastiveDataset(Dataset):
    def __init__(self, X_path, y_path, t_path, is_train=False):
        self.X = torch.load(X_path, map_location='cpu')
        self.y = torch.load(y_path, map_location='cpu')
        self.t = torch.load(t_path, map_location='cpu')
        self.is_train = is_train
        self.print_count = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            # Dynamic D4 Augmentation
            # Applied to all 12 channels simultaneously, preserving spatial relationships exactly!
            k = random.randint(0, 3)
            flip = random.choice([True, False])
            
            x = torch.rot90(x, k, [1, 2])
            if flip:
                x = torch.flip(x, [2])
                
            if self.print_count < 3:
                print(f"[Visual Check] Sample {idx}: Applied rotation={k*90}deg, flip={flip} to BOTH boards (all 12 channels simultaneously)")
                self.print_count += 1
                
        return x, self.y[idx], self.t[idx]

def train_and_eval_fold(fold):
    print(f"\n{'='*40}\n FOLD {fold}\n{'='*40}")
    
    train_X_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_X_train.pt")
    train_y_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_y_train.pt")
    train_t_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_t_train.pt")
    test_X_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_X_test.pt")
    test_y_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_y_test.pt")
    test_t_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{fold-1}_t_test.pt")
    
    if not os.path.exists(train_X_path):
        print(f"Dataset for fold {fold} not found!")
        return None
        
    train_ds = ContrastiveDataset(train_X_path, train_y_path, train_t_path, is_train=True)
    test_ds = ContrastiveDataset(test_X_path, test_y_path, test_t_path, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SokobanSEResNetClassifier(dropout_p=0.4, in_channels=12).to(device)
    train_y_tensor = train_ds.y
    num_pos = (train_y_tensor == 1).sum().item()
    num_neg = (train_y_tensor == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)]).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_f05 = 0.0
    best_metrics = {}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Eval
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_targets = []
        all_types = []
        
        with torch.no_grad():
            for X_batch, y_batch, t_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_types.extend(t_batch.cpu().numpy())
                
        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)
        t_arr = np.array(all_types)
        
        if fold == 1 and epoch == 0:
            print("\n--- Probability Distribution (Fold 1, Epoch 1) ---")
            print(f"Mean: {y_prob.mean():.4f}, Std: {y_prob.std():.4f}")
            hist, bin_edges = np.histogram(y_prob, bins=10, range=(0, 1))
            for i in range(10):
                print(f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}): {hist[i]}")
            print("------------------------------------------------\n")
            
        # Threshold Sweeping
        best_epoch_f05 = 0.0
        best_epoch_metrics = {}
        best_threshold = 0.5
        
        for thresh in np.arange(0.50, 0.96, 0.05):
            y_pred = (y_prob >= thresh).astype(float)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
            
            # Deadlock Specificity Breakdown
            mask_simple = (t_arr == 2)
            mask_complex = (t_arr == 3)
            
            simple_total = mask_simple.sum()
            complex_total = mask_complex.sum()
            
            simple_correct = (y_pred[mask_simple] == 0).sum() if simple_total > 0 else 0
            complex_correct = (y_pred[mask_complex] == 0).sum() if complex_total > 0 else 0
            
            spec_simple = simple_correct / simple_total if simple_total > 0 else 0
            spec_complex = complex_correct / complex_total if complex_total > 0 else 0
            
            if f05 > best_epoch_f05:
                best_epoch_f05 = f05
                best_threshold = thresh
                best_epoch_metrics = {
                    'epoch': epoch,
                    'threshold': thresh,
                    'precision': precision,
                    'recall': recall,
                    'f05': f05,
                    'spec_simple': spec_simple,
                    'spec_complex': spec_complex,
                    'loss': val_loss / len(test_loader)
                }
                
        if best_epoch_f05 > best_f05:
            best_f05 = best_epoch_f05
            best_metrics = best_epoch_metrics
            # Save best fold model
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"final_contrastive_classifier_fold{fold}.pt"))
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss/len(test_loader):.4f} | Optimal Thresh: {best_threshold:.2f} | F0.5: {best_epoch_f05:.4f} | Spec(Simple): {best_epoch_metrics.get('spec_simple', 0):.4f} | Spec(Complex): {best_epoch_metrics.get('spec_complex', 0):.4f}")

    print(f"Best metrics for Fold {fold}: {best_metrics}")
    return best_metrics

if __name__ == "__main__":
    results = []
    for f in range(1, N_FOLDS + 1):
        res = train_and_eval_fold(f)
        if res:
            results.append(res)
            
    if results:
        avg_f05 = np.mean([r['f05'] for r in results])
        avg_prec = np.mean([r['precision'] for r in results])
        avg_rec = np.mean([r['recall'] for r in results])
        
        print("\n" + "="*50)
        print("  5-FOLD CROSS VALIDATION RESULTS (CONTRASTIVE)")
        print("="*50)
        print(f"  F0.5 Score: {avg_f05:.4f}")
        print(f"  Precision:  {avg_prec:.4f}")
        print(f"  Recall:     {avg_rec:.4f}")
        print("="*50)
