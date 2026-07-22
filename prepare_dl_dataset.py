import os
import glob
import re
import hashlib
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from tqdm import tqdm

DATA_DIR = "/mnt/c/Users/hanss/Desktop/backup oficina/antes de que se corte la luz 1 preventivo/todos"
MAX_SAMPLES_PER_BUCKET = 1000

print("="*50)
print(" PREPARANDO DATASET DE DEEP LEARNING (TENSOR 5C)")
print("="*50)

def get_bucket(pushes):
    if pushes <= 10: return "1_to_10"
    if pushes > 100: return "101_plus"
    lower = ((pushes - 1) // 10) * 10 + 1
    upper = lower + 9
    return f"{lower}_to_{upper}"

def parse_sok_files(directory):
    records = []
    files = glob.glob(os.path.join(directory, "**/*.sok"), recursive=True)
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3: continue
            header = lines[0]
            board_lines = lines[1:]
            
            m = re.match(r"(\d+)\s*-\s*(.*)", header)
            if not m: continue
            stats_str = m.group(2)
            
            pushes = -1
            if "pushes:" in stats_str:
                for token in stats_str.split():
                    if token.startswith("pushes:"):
                        pushes = int(token.split(":")[1])
            elif stats_str.isdigit():
                pushes = int(stats_str)
            
            if pushes <= 0: continue
            
            board_str = "\n".join(board_lines)
            board_hash = hashlib.sha256(board_str.encode()).hexdigest()
            records.append({
                "hash": board_hash,
                "board_str": board_str,
                "pushes": pushes,
                "bucket": get_bucket(pushes)
            })
    return records

print("1. Parseando archivos .sok...")
all_records = parse_sok_files(DATA_DIR)
df = pd.DataFrame(all_records)
df = df.drop_duplicates(subset=["hash"])
print(f"Tableros unicos encontrados: {len(df)}")

print("\n2. Balanceando dataset (Max 1000 por cubeta)...")
balanced_df = pd.concat([group.sample(min(len(group), MAX_SAMPLES_PER_BUCKET), random_state=42) for _, group in df.groupby('bucket')])
print("Conteo por cubetas:")
print(balanced_df['bucket'].value_counts())
print(f"Total tras balanceo: {len(balanced_df)}")

print("\n3. Configurando Funciones de Conversión a Tensores (K-Fold CV)...")

# --- CONVERSION A TENSORES ---

def flood_fill_exterior(char_matrix):
    H, W = char_matrix.shape
    visited = np.zeros((H, W), dtype=bool)
    q = []
    # Iniciar bordes
    for r in range(H):
        q.append((r, 0)); q.append((r, W-1))
    for c in range(W):
        q.append((0, c)); q.append((H-1, c))
    
    exterior = np.zeros((H, W), dtype=bool)
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        if r < 0 or r >= H or c < 0 or c >= W: continue
        if visited[r, c]: continue
        visited[r, c] = True
        
        if char_matrix[r, c] == '#': continue
            
        exterior[r, c] = True
        q.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
    return exterior

def encode_board(board_str):
    lines = board_str.splitlines()
    H = len(lines)
    W = max(len(l) for l in lines)
    
    char_matrix = np.full((H, W), ' ', dtype=str)
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            char_matrix[r, c] = char
            
    exterior = flood_fill_exterior(char_matrix)
    tensor = np.zeros((5, H, W), dtype=np.float32)
    
    for r in range(H):
        for c in range(W):
            ch = char_matrix[r, c]
            if ch == '#':
                tensor[0, r, c] = 1.0 # Canal 0: Pared
            if not exterior[r, c] and ch != '#':
                tensor[1, r, c] = 1.0 # Canal 1: Interior caminable
            if ch in ['$', '*']:
                tensor[2, r, c] = 1.0 # Canal 2: Cajas
            if ch in ['.', '*', '+']:
                tensor[3, r, c] = 1.0 # Canal 3: Metas
            if ch in ['@', '+']:
                tensor[4, r, c] = 1.0 # Canal 4: Jugador
                
    return tensor

def augment_d4(tensor):
    augmented = []
    for k in range(4):
        rot = np.rot90(tensor, k=k, axes=(1, 2))
        augmented.append(rot.copy())
        refl = np.flip(rot, axis=2) # Reflexion
        augmented.append(refl.copy())
    return augmented

print("\n4. Ejecutando Stratified K-Fold CV (5 Folds)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(balanced_df, balanced_df['bucket'])):
    print(f"\n=============================================")
    print(f" PROCESANDO FOLD {fold + 1} / 5")
    print(f"=============================================")
    
    train_df = balanced_df.iloc[train_idx]
    test_df = balanced_df.iloc[test_idx]
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    print(f"\nFold {fold+1}: Codificando TEST SET (Sin Augmentation)...")
    test_data = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        t = encode_board(row['board_str'])
        test_data.append({"tensor": torch.tensor(t), "pushes": row['pushes']})

    print(f"\nFold {fold+1}: Codificando TRAIN SET + D4 Augmentation...")
    train_data = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        t = encode_board(row['board_str'])
        # Aplicar D4 (8 variantes por cada tablero original)
        t_variants = augment_d4(t)
        for v in t_variants:
            train_data.append({"tensor": torch.tensor(v), "pushes": row['pushes']})

    print(f"\nFold {fold+1} Tamaño final Train (x8 debido a D4): {len(train_data)}")
    print(f"Fold {fold+1} Tamaño final Test (x1 original):     {len(test_data)}")

    print(f"\nFold {fold+1}: Guardando tensores de PyTorch (.pt)...")
    torch.save(train_data, f"dl_dataset_fold{fold+1}_train.pt")
    torch.save(test_data, f"dl_dataset_fold{fold+1}_test.pt")
    
print("\n✅ Proceso de Cross Validation completado exitosamente.")
