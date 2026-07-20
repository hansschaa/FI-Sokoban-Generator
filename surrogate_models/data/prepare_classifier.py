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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVABLES_DIR = os.path.join(BASE_DIR, "..", "..", "training_data", "Solvables")
UNSOLVABLES_FILE = os.path.join(BASE_DIR, "..", "..", "training_data", "Unsolvables", "deadlocks.sok")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 5
MAX_H = 25
MAX_W = 25

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO
# ─────────────────────────────────────────────────────────────────────────────
def parse_sok_file(fpath, default_label=None):
    records = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        
        board_str = "\n".join(lines[1:])
        MOBILE_CHARS = str.maketrans("$.*@+", "     ")
        shell_str = board_str.translate(MOBILE_CHARS)
        shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()
        board_hash = hashlib.sha256(board_str.encode()).hexdigest()

        records.append({
            "board_hash": board_hash,
            "shell_hash": shell_hash,
            "board_str": board_str,
            "label": default_label
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# CODIFICACION
# ─────────────────────────────────────────────────────────────────────────────
def encode_board(board_str, max_h=MAX_H, max_w=MAX_W):
    lines = board_str.splitlines()
    H, W = len(lines), max(len(l) for l in lines) if lines else 0
    H, W = min(H, max_h), min(W, max_w)

    char_matrix = np.full((H, W), ' ', dtype=str)
    for r, line in enumerate(lines[:H]):
        for c, ch in enumerate(line[:W]):
            char_matrix[r, c] = ch

    tensor = np.zeros((5, max_h, max_w), dtype=np.float32)
    tensor[0, :, :] = 1.0  # Fondo es muro

    # Centrado
    offset_r = (max_h - H) // 2
    offset_c = (max_w - W) // 2

    for r in range(H):
        for c in range(W):
            ch = char_matrix[r, c]
            rr, cc = r + offset_r, c + offset_c
            tensor[0, rr, cc] = 0.0

            if ch == '#':
                tensor[0, rr, cc] = 1.0
            elif ch == ' ':
                tensor[1, rr, cc] = 1.0
            elif ch == '$':
                tensor[1, rr, cc] = 1.0
                tensor[2, rr, cc] = 1.0
            elif ch == '.':
                tensor[1, rr, cc] = 1.0
                tensor[3, rr, cc] = 1.0
            elif ch == '*':
                tensor[1, rr, cc] = 1.0
                tensor[2, rr, cc] = 1.0
                tensor[3, rr, cc] = 1.0
            elif ch == '@':
                tensor[1, rr, cc] = 1.0
                tensor[4, rr, cc] = 1.0
            elif ch == '+':
                tensor[1, rr, cc] = 1.0
                tensor[3, rr, cc] = 1.0
                tensor[4, rr, cc] = 1.0
            else:
                tensor[1, rr, cc] = 1.0
    return tensor

def augment_tensor(tensor):
    variants = [tensor]
    variants.append(np.flip(tensor, axis=1)) # Vertical
    variants.append(np.flip(tensor, axis=2)) # Horizontal
    variants.append(np.flip(np.flip(tensor, axis=1), axis=2)) # Rot 180
    
    t_T = np.transpose(tensor, axes=(0, 2, 1))
    variants.append(t_T)
    variants.append(np.flip(t_T, axis=1))
    variants.append(np.flip(t_T, axis=2))
    variants.append(np.flip(np.flip(t_T, axis=1), axis=2))
    return variants

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("1. Cargando y parseando Solubles (label=1)...")
    solvables = []
    solvable_files = glob.glob(os.path.join(SOLVABLES_DIR, "*", "*.sok"))
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
    # Raro, pero posible si el solver marcó un deadlock como soluble por bug
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
        
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        print("  Generando Train con Augmentation (x8)...")
        train_data = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
            t = encode_board(row["board_str"])
            for v in augment_tensor(t):
                train_data.append({
                    "tensor": torch.from_numpy(v.copy()),
                    "label": row["label"]
                })
                
        print("  Generando Test (sin Augmentation)...")
        test_data = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            test_data.append({
                "tensor": torch.from_numpy(encode_board(row["board_str"])),
                "label": row["label"]
            })

        print(f"  Train tensors: {len(train_data):,} | Test tensors: {len(test_data):,}")
        
        train_path = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_train.pt")
        test_path  = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")
        
        torch.save(train_data, train_path)
        torch.save(test_data, test_path)
        print(f"  ✅ Guardados fold {fold} en results/")

if __name__ == "__main__":
    main()
