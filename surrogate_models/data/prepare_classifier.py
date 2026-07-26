"""
prepare_classifier.py
---------------------
Parsea tableros solubles (label 1) y deadlocks (label 0).
Calcula shell_hash y aplica GroupKFold(5) estratificado por clase (0/1).
Aplica Data Augmentation D4 (x8) SOLO al conjunto de Train.
Guarda los 5 pares (train/test) en results/.
"""

import os
import glob
import hashlib
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from data.board_utils import encode_board, augment_tensor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVABLES_DIR = os.path.join(BASE_DIR, "..", "..", "training_data", "Solvables")
UNSOLVABLES_FILE = os.path.join(BASE_DIR, "..", "..", "training_data", "Unsolvables", "deadlocks.sok")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 5
do_augmentation = True

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO
# ─────────────────────────────────────────────────────────────────────────────
def parse_sok_file(fpath, default_label=None):
    records = []
    import re
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        
        header = lines[0]
        board_str = "\n".join(lines[1:])
        
        # Parse metadata from header if available
        dtype = "SOLVABLE"
        mutations = 0
        source_hash = ""
        
        if default_label == 0:
            m_type = re.search(r"type:([A-Z_]+)", header)
            if m_type: dtype = m_type.group(1)
            
            m_mut = re.search(r"mutations:(\d+)", header)
            if m_mut: mutations = int(m_mut.group(1))
            
            m_src = re.search(r"source_hash:([0-9a-fA-F\-]+)", header)
            if m_src: source_hash = m_src.group(1)

        MOBILE_CHARS = str.maketrans("$.*@+", "     ")
        shell_str = board_str.translate(MOBILE_CHARS)
        shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()
        board_hash = hashlib.sha256(board_str.encode()).hexdigest()
        
        # Fallback for source_hash if not provided in header
        if not source_hash:
            source_hash = shell_hash

        records.append({
            "board_hash": board_hash,
            "shell_hash": shell_hash,
            "source_hash": source_hash,
            "board_str": board_str,
            "label": default_label,
            "deadlock_type": dtype,
            "mutations": mutations
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("1. Cargando y parseando Solubles (label=1)...")
    solvables = []
    solvable_files = glob.glob(os.path.join(SOLVABLES_DIR, "**", "*.sok"), recursive=True)
    for f in tqdm(solvable_files):
        solvables.extend(parse_sok_file(f, default_label=1))
        
    print("2. Cargando y parseando Deadlocks (label=0)...")
    unsolvables = parse_sok_file(UNSOLVABLES_FILE, default_label=0)
    
    all_records = solvables + unsolvables
    
    # Deduplicar por board_hash
    df = pd.DataFrame(all_records)
    print(f"Total antes de deduplicar: {len(df)}")
    df.drop_duplicates(subset=["board_hash"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Total tras deduplicar exacto: {len(df)}")
    
    # Eliminar boards que existan en AMBAS clases (leakage semántico fatal)
    hash_counts = df.groupby('board_hash')['label'].nunique()
    conflict_hashes = hash_counts[hash_counts > 1].index
    if len(conflict_hashes) > 0:
        print(f"⚠️ ADVERTENCIA: {len(conflict_hashes)} tableros con label en conflicto. Eliminándolos.")
        df = df[~df['board_hash'].isin(conflict_hashes)].reset_index(drop=True)

    print(f"Distribución Final: Solubles={sum(df['label']==1)}, Deadlocks={sum(df['label']==0)}")

    X = df.index.values
    y = df['label'].values
    groups = df['shell_hash'].values

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        fold += 1
        print(f"\n[{'='*40}]")
        print(f" Procesando FOLD {fold}/{N_FOLDS}")
        print(f"[{'='*40}]")
        
        train_df_full = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx]
        
        # Nested split para Validation (20% del Train original)
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        val_splits = list(sgkf_val.split(train_df_full.index.values, train_df_full['label'].values, train_df_full['shell_hash'].values))
        train_final_idx, val_idx = val_splits[0]
        
        train_df = train_df_full.iloc[train_final_idx]
        val_df   = train_df_full.iloc[val_idx]

        def allocate_and_fill_direct(df, do_aug):
            aug_factor = 8 if do_aug else 1
            N = len(df) * aug_factor
            
            tensors = torch.empty((N, 6, 25, 25), dtype=torch.float32)
            is_solvable = torch.empty(N, dtype=torch.uint8)
            deadlock_type = []
            mutations = torch.empty(N, dtype=torch.uint8)
            source_hash = []
            
            idx = 0
            for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
                t_np = encode_board(row["board_str"])
                if do_aug:
                    for t_aug in augment_tensor(t_np):
                        tensors[idx] = torch.from_numpy(t_aug.copy())
                        is_solvable[idx] = row["label"]
                        deadlock_type.append(row["deadlock_type"])
                        mutations[idx] = row["mutations"]
                        source_hash.append(row["source_hash"])
                        idx += 1
                else:
                    tensors[idx] = torch.from_numpy(t_np)
                    is_solvable[idx] = row["label"]
                    deadlock_type.append(row["deadlock_type"])
                    mutations[idx] = row["mutations"]
                    source_hash.append(row["source_hash"])
                    idx += 1
                    
            return {
                "tensor": tensors,
                "is_solvable": is_solvable,
                "deadlock_type": deadlock_type,
                "mutations": mutations,
                "source_hash": source_hash,
            }

        print(f"  Generando Train con Augmentation...")
        train_data = allocate_and_fill_direct(train_df, do_augmentation)
        
        print("  Generando Validation (sin Augmentation)...")
        val_data = allocate_and_fill_direct(val_df, False)
        
        print("  Generando Test (sin Augmentation)...")
        test_data = allocate_and_fill_direct(test_df, False)
        
        print(f"  Train: {len(train_data['is_solvable']):,} | Val: {len(val_data['is_solvable']):,} | Test: {len(test_data['is_solvable']):,}")
        
        train_path = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_train.pt")
        val_path   = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_val.pt")
        test_path  = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")
        
        torch.save(train_data, train_path, _use_new_zipfile_serialization=False)
        torch.save(val_data, val_path, _use_new_zipfile_serialization=False)
        torch.save(test_data, test_path, _use_new_zipfile_serialization=False)
        print(f"  ✅ Guardados fold {fold} (Train/Val/Test) en results/")

if __name__ == "__main__":
    main()
