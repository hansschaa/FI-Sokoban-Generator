import sys, os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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
            b_type = 1
            for l in lines:
                if l.startswith("label:"):
                    label = int(l.split(":")[1])
                elif l.startswith("type:"):
                    t_str = l.split(":")[1].strip()
                    if t_str == "simple": b_type = 2
                    elif t_str == "complex": b_type = 3
            
            source_board = "\n".join(lines[source_idx+1:mutated_idx])
            
            import hashlib
            shell_hash = hashlib.sha256(source_board.encode()).hexdigest()
            
            records.append({
                "label": label,
                "type": b_type,
                "shell_hash": shell_hash
            })
        except ValueError:
            continue
    return records

def main():
    print("Fast-generating s_train and s_test tensors...")
    
    label_0_file = os.path.join(BASE_DIR, "..", "training_data", "ContrastivePairs", "label_0_deadlocks.sok")
    label_1_file = os.path.join(BASE_DIR, "..", "training_data", "ContrastivePairs", "label_1_solvables.sok")
    
    dense_label_0_file = os.path.join(BASE_DIR, "..", "training_data", "DenseContrastivePairs", "label_0_deadlocks.sok")
    dense_label_1_file = os.path.join(BASE_DIR, "..", "training_data", "DenseContrastivePairs", "label_1_solvables.sok")
    
    records_orig = []
    if os.path.exists(label_0_file): records_orig.extend(parse_contrastive_sok_file(label_0_file))
    if os.path.exists(label_1_file): records_orig.extend(parse_contrastive_sok_file(label_1_file))
        
    records_dense = []
    if os.path.exists(dense_label_0_file): records_dense.extend(parse_contrastive_sok_file(dense_label_0_file))
    if os.path.exists(dense_label_1_file): records_dense.extend(parse_contrastive_sok_file(dense_label_1_file))
    
    df_orig = pd.DataFrame(records_orig)
    df_orig['source_dataset'] = 'original'
    
    df_dense = pd.DataFrame(records_dense)
    if len(df_dense) > 0:
        df_dense['source_dataset'] = 'dense'
    else:
        df_dense = pd.DataFrame(columns=df_orig.columns)
        df_dense['source_dataset'] = []
        
    if len(df_dense) > 0 and len(df_dense) < len(df_orig):
        df_dense = df_dense.sample(n=len(df_orig), replace=True, random_state=42).reset_index(drop=True)
    elif len(df_orig) > 0 and len(df_orig) < len(df_dense):
        df_orig = df_orig.sample(n=len(df_dense), replace=True, random_state=42).reset_index(drop=True)
        
    df = pd.concat([df_orig, df_dense], ignore_index=True)
    
    df_label_0 = df[df['label'] == 0]
    df_label_1 = df[df['label'] == 1]
    
    if len(df_label_0) > len(df_label_1) and len(df_label_1) > 0:
        df_label_1 = df_label_1.sample(n=len(df_label_0), replace=True, random_state=42)
    elif len(df_label_1) > len(df_label_0) and len(df_label_0) > 0:
        df_label_0 = df_label_0.sample(n=len(df_label_1), replace=True, random_state=42)
        
    df = pd.concat([df_label_0, df_label_1], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['source_int'] = df['source_dataset'].apply(lambda x: 0 if x == 'original' else 1)
    
    y = np.array(df['label'].values, dtype=np.float32)
    s = np.array(df['source_int'].values, dtype=np.int64)
    groups = df['shell_hash'].values
    X_dummy = np.zeros(len(y))
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_dummy, y, groups)):
        s_train, s_test = s[train_idx], s[test_idx]
        torch.save(torch.from_numpy(s_train).long(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_s_train.pt"))
        torch.save(torch.from_numpy(s_test).long(), os.path.join(RESULTS_DIR, f"contrastive_fold_{fold}_s_test.pt"))
        print(f"Generated fold {fold} s_train/s_test.")
        
if __name__ == "__main__":
    main()
