import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.metrics import fbeta_score, precision_score, recall_score

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FOLD = 1
MAX_EPOCHS = 15
BETA = 0.5

sys.path.insert(0, BASE_DIR)
from models.resnet import SokobanSEResNetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db_url     = os.environ.get("OPTUNA_DB_URL", f"sqlite:///{RESULTS_DIR}/optuna_contrastive_classifier.db")
study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_contrastive_lab_v3")

class ContrastiveMemoryDataset(Dataset):
    def __init__(self, X_tensor, y_tensor, t_tensor, s_tensor=None):
        self.X = X_tensor
        self.y = y_tensor
        self.t = t_tensor
        self.s = s_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.s is None:
            import random
            k = random.randint(0, 3)
            flip = random.choice([True, False])
            x = torch.rot90(x, k, [1, 2])
            if flip:
                x = torch.flip(x, [2])
            return x, self.y[idx], self.t[idx]
        else:
            return x, self.y[idx], self.t[idx], self.s[idx]

def load_data():
    train_X = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_X_train.pt"), map_location='cpu')
    train_y = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_y_train.pt"), map_location='cpu')
    train_t = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_t_train.pt"), map_location='cpu')
    
    val_X = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_X_test.pt"), map_location='cpu')
    val_y = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_y_test.pt"), map_location='cpu')
    val_t = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_t_test.pt"), map_location='cpu')
    val_s = torch.load(os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_s_test.pt"), map_location='cpu')
    
    num_pos = (train_y == 1).sum().item()
    num_neg = (train_y == 0).sum().item()
    pos_weight = num_neg / max(1, num_pos)
    
    return train_X, train_y, train_t, val_X, val_y, val_t, val_s, pos_weight

def train_and_evaluate(trial, train_X, train_y, train_t, val_X, val_y, val_t, val_s, pos_weight):
    lr = trial.params['lr']
    weight_decay = trial.params['weight_decay']
    dropout_p = trial.params['dropout_p']
    batch_size = trial.params['batch_size']
    
    print(f"\n--- Retraining Trial {trial.number} (Global F0.5: {trial.value:.4f}) ---")
    print(f"Params: lr={lr:.5f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, bs={batch_size}")
    
    train_dataset = ContrastiveMemoryDataset(train_X, train_y, train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ContrastiveMemoryDataset(val_X, val_y, val_t, val_s)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SokobanSEResNetClassifier(dropout_p=dropout_p, in_channels=12).to(device)
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    best_dense_f05 = 0.0
    best_dense_stats = {}

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
        
        # Eval
        model.eval()
        all_probs, all_targets, all_t, all_s = [], [], [], []
        with torch.no_grad():
            for X_batch, y_batch, t_batch, s_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(y_batch.numpy())
                all_t.extend(t_batch.numpy())
                all_s.extend(s_batch.numpy())
                
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        all_t = np.array(all_t)
        all_s = np.array(all_s)
        
        # Filter to dense subset (s == 1)
        dense_mask = (all_s == 1)
        dense_probs = all_probs[dense_mask]
        dense_targets = all_targets[dense_mask]
        dense_t = all_t[dense_mask]
        
        best_epoch_dense_f05 = 0.0
        epoch_stats = {}
        for thresh in np.arange(0.50, 0.96, 0.05):
            preds = (dense_probs >= thresh).astype(float)
            fb = fbeta_score(dense_targets, preds, beta=BETA, zero_division=0)
            if fb > best_epoch_dense_f05:
                best_epoch_dense_f05 = fb
                
                # Spec breakdown
                # simple deadlocks (type == 2) -> target is 0, so we want recall of class 0, which is spec/TN
                # wait, precision is TP / (TP+FP). recall is TP / (TP+FN).
                # To break down TN rate (spec) by simple/complex:
                simple_mask = (dense_t == 2)
                complex_mask = (dense_t == 3)
                
                # Spec = TN / (TN + FP). Here Negatives are deadlocks.
                # true negatives are those where pred == 0 and target == 0
                if np.sum(simple_mask) > 0:
                    spec_simple = np.sum((preds[simple_mask] == 0) & (dense_targets[simple_mask] == 0)) / np.sum(simple_mask)
                else:
                    spec_simple = 0.0
                    
                if np.sum(complex_mask) > 0:
                    spec_complex = np.sum((preds[complex_mask] == 0) & (dense_targets[complex_mask] == 0)) / np.sum(complex_mask)
                else:
                    spec_complex = 0.0
                    
                epoch_stats = {
                    "thresh": thresh,
                    "f05": fb,
                    "spec_simple": spec_simple,
                    "spec_complex": spec_complex
                }
                
        if best_epoch_dense_f05 > best_dense_f05:
            best_dense_f05 = best_epoch_dense_f05
            best_dense_stats = epoch_stats

    return best_dense_stats

def main():
    print("Loading Optuna Study...")
    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        return
        
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    top_k = 5
    top_trials = completed_trials[:top_k]
    
    if not top_trials:
        print("No completed trials found.")
        return
        
    print(f"Top {len(top_trials)} trials by Global F0.5:")
    for t in top_trials:
        print(f"  Trial {t.number}: Global F0.5 = {t.value:.4f}")
        
    print("\nLoading datasets...")
    train_X, train_y, train_t, val_X, val_y, val_t, val_s, pos_weight = load_data()
    
    results = []
    for t in top_trials:
        stats = train_and_evaluate(t, train_X, train_y, train_t, val_X, val_y, val_t, val_s, pos_weight)
        print(f"--> Trial {t.number} Dense Eval: F0.5={stats['f05']:.4f} @ thr={stats['thresh']:.2f}, "
              f"Spec Simple={stats['spec_simple']:.4f}, Spec Complex={stats['spec_complex']:.4f}")
        results.append({
            "trial": t.number,
            "dense_f05": stats['f05'],
            "spec_simple": stats['spec_simple'],
            "spec_complex": stats['spec_complex'],
            "thresh": stats['thresh'],
            "global_f05": t.value
        })
        
    print("\n--- FINAL RANKING ON DENSE SUBSET ---")
    results.sort(key=lambda x: x['dense_f05'], reverse=True)
    for r in results:
        print(f"Trial {r['trial']}: Dense F0.5 = {r['dense_f05']:.4f} (Global F0.5: {r['global_f05']:.4f}) | "
              f"Spec Simp={r['spec_simple']:.4f}, Spec Comp={r['spec_complex']:.4f}")

if __name__ == "__main__":
    main()
