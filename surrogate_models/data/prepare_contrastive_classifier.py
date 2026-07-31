import os
import glob
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from data.board_utils import encode_board, augment_tensor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 5
do_augmentation = False

def parse_contrastive_sok_file(fpath):
    records = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        
        try:
            source_idx = lines.index("source_board:")
            mutated_idx = lines.index("mutated_board:")
            
            label = 0
            b_type = 1 # 1=Solvable, 2=Simple Deadlock, 3=Complex Deadlock
            for l in lines:
                if l.startswith("label:"):
                    label = int(l.split(":")[1])
                elif l.startswith("type:"):
                    t_str = l.split(":")[1].strip()
                    if t_str == "simple":
                        b_type = 2
                    elif t_str == "complex":
                        b_type = 3
            
            source_board = "\n".join(lines[source_idx+1:mutated_idx])
            
            mutated_end_idx = len(lines)
            for i in range(mutated_idx+1, len(lines)):
                if lines[i].startswith("label:") or lines[i].startswith("pushes:"):
                    mutated_end_idx = i
                    break
            mutated_board = "\n".join(lines[mutated_idx+1:mutated_end_idx])
            
            import hashlib
            shell_hash = hashlib.sha256(source_board.encode()).hexdigest()
            
            records.append({
                "source_board": source_board,
                "mutated_board": mutated_board,
                "label": label,
                "type": b_type,
                "shell_hash": shell_hash
            })
        except ValueError:
            continue
            
    return records

def main():
    print("Loading contrastive dataset...")
    records = []
    
    label_0_file = os.path.join(BASE_DIR, "..", "..", "training_data", "ContrastivePairs", "label_0_deadlocks.sok")
    label_1_file = os.path.join(BASE_DIR, "..", "..", "training_data", "ContrastivePairs", "label_1_solvables.sok")
    
    if os.path.exists(label_0_file):
        records.extend(parse_contrastive_sok_file(label_0_file))
    if os.path.exists(label_1_file):
        records.extend(parse_contrastive_sok_file(label_1_file))
        
    df = pd.DataFrame(records)
    print(f"Total pairs loaded: {len(df)}")
    if len(df) == 0:
        print("No data found!")
        return
        
    if len(df) < 100000:
        print("\n" + "!"*60)
        print("WARNING: Less than 100,000 pairs loaded!")
        print("Did you forget to combine the .sok files from all 3 PCs?")
        print("Expected ~105,000 pairs, but found:", len(df))
        print("!"*60 + "\n")
        
    print(df['label'].value_counts())

    print("Encoding boards (12-channel tensors)...")
    tensors = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            source_t = encode_board(row['source_board'])
            mutated_t = encode_board(row['mutated_board'])
            combined_t = np.concatenate([source_t, mutated_t], axis=0)
            tensors.append(combined_t)
            labels.append(row['label'])
        except Exception as e:
            print(f"Error encoding row {idx}: {e}")
            tensors.append(None)
            
    df['tensor'] = tensors
    df['label'] = labels
    df = df.dropna(subset=['tensor'])
    
    X = np.stack(df['tensor'].values)
    y = np.array(df['label'].values, dtype=np.float32)
    t = np.array(df['type'].values, dtype=np.int64)
    groups = df['shell_hash'].values
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}, t={t.shape}")
    
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"--- Processing FOLD {fold+1}/{N_FOLDS} ---")
        
        X_train, y_train, t_train = X[train_idx], y[train_idx], t[train_idx]
        X_test, y_test, t_test   = X[test_idx], y[test_idx], t[test_idx]
        
        if do_augmentation:
            X_train_aug = []
            y_train_aug = []
            for i in range(len(X_train)):
                t_aug = augment_tensor(X_train[i])
                for ta in t_aug:
                    X_train_aug.append(ta)
                    y_train_aug.append(y_train[i])
            X_train = np.stack(X_train_aug)
            y_train = np.array(y_train_aug, dtype=np.float32)
            # t_train is NOT augmented currently, if do_augmentation is true, it won't match. But do_augmentation is False now.
            
        torch.save(torch.from_numpy(X_train).float(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_X_train.pt"))
        torch.save(torch.from_numpy(y_train).float(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_y_train.pt"))
        torch.save(torch.from_numpy(t_train).long(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_t_train.pt"))
        torch.save(torch.from_numpy(X_test).float(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_X_test.pt"))
        torch.save(torch.from_numpy(y_test).float(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_y_test.pt"))
        torch.save(torch.from_numpy(t_test).long(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_t_test.pt"))
        
        print(f"Fold {fold} saved. Train: {X_train.shape}, Test: {X_test.shape}")
        
if __name__ == "__main__":
    main()
