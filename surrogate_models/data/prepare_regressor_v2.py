"""
prepare_regressor_v2.py
--------------------
Parsea tableros de Solvables/ (Originales Anchos) y DenseSolvables/ (Densos).
Verifica el balance por origen y bucket.
Aplica GroupKFold(5) estratificado por origen+bucket y guarda tensores.
"""

import os
import re
import glob
import hashlib
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from data.board_utils import encode_board, augment_tensor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVABLES_DIR = os.path.join(BASE_DIR, "..", "..", "training_data", "Solvables")
DENSE_SOLVABLES_DIR = os.path.join(BASE_DIR, "..", "..", "training_data", "DenseSolvables")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 5
MAX_H = 25
MAX_W = 25

def parse_sok_file(fpath, source_label):
    records = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2: continue

        header = lines[0]
        board_lines = lines[1:]

        pushes = None
        m_pushes = re.search(r"pushes:(\d+)", header)
        if m_pushes: pushes = int(m_pushes.group(1))

        if pushes is None or pushes <= 0: continue

        board_str = "\n".join(board_lines)

        MOBILE_CHARS = str.maketrans("$.*@+", "     ")
        shell_str = board_str.translate(MOBILE_CHARS)
        shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()
        board_hash = hashlib.sha256(board_str.encode()).hexdigest()

        if pushes <= 10: bucket = "1_to_10"
        elif pushes > 100: bucket = "101_plus"
        else:
            lower = ((pushes - 1) // 10) * 10 + 1
            bucket = f"{lower}_to_{lower+9}"

        box_count = board_str.count('$') + board_str.count('*')

        records.append({
            "board_str": board_str,
            "pushes": pushes,
            "bucket": bucket,
            "shell_hash": shell_hash,
            "board_hash": board_hash,
            "box_count": box_count,
            "source": source_label
        })
    return records

def main():
    print("=" * 70)
    print("  PREPARANDO DATASET REGRESOR V2 (Original + Denso)")
    print("=" * 70)

    all_records = []
    
    wide_files = glob.glob(os.path.join(SOLVABLES_DIR, "**", "*.sok"), recursive=True)
    for fpath in tqdm(wide_files, desc="Parseando Solvables (Original)"):
        all_records.extend(parse_sok_file(fpath, "Original"))
        
    dense_files = glob.glob(os.path.join(DENSE_SOLVABLES_DIR, "**", "*.sok"), recursive=True)
    for fpath in tqdm(dense_files, desc="Parseando DenseSolvables"):
        all_records.extend(parse_sok_file(fpath, "Denso"))

    df = pd.DataFrame(all_records)
    print(f"\n[1] Total parseado: {len(df)} tableros")

    df = df.drop_duplicates(subset=["board_hash"]).reset_index(drop=True)
    print(f"[2] Tras deduplicar: {len(df)} tableros")

    print("\n[3] 🛑 CHEQUEO CRÍTICO DE BALANCE (Origen vs Bucket) 🛑")
    balance_df = pd.crosstab(df['bucket'], df['source'], margins=True)
    print(balance_df.to_string())
    
    print("\n[!] POR FAVOR REVISAR LA TABLA ARRIBA ANTES DE CONTINUAR EL ENTRENAMIENTO.")
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["shell_hash"].values
    fold_splits = list(gkf.split(df, groups=groups))

    print(f"\n[4] Codificando tensores y aplicando Data Augmentation...")
    for fold_idx, (outer_train_idx, test_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        print(f"\n  ── FOLD {fold_num}/{N_FOLDS} ──────────────────────────────")

        outer_train_df = df.iloc[outer_train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx]

        inner_gkf = GroupKFold(n_splits=5)
        inner_groups = outer_train_df["shell_hash"].values
        inner_train_idx, val_idx = next(inner_gkf.split(outer_train_df, groups=inner_groups))
        
        train_df = outer_train_df.iloc[inner_train_idx]
        val_df = outer_train_df.iloc[val_idx]

        pushes_train = train_df["pushes"].values.astype(float)
        pushes_log = np.log1p(pushes_train)
        pushes_mean = pushes_log.mean()
        pushes_std = pushes_log.std()

        box_counts = train_df["box_count"].value_counts()
        total_train = len(train_df)
        class_weights = {k: total_train / (len(box_counts) * v) for k, v in box_counts.items()}

        stats = {"pushes_mean": float(pushes_mean), "pushes_std": float(pushes_std)}
        
        test_data = []
        for _, row in test_df.iterrows():
            t = encode_board(row["board_str"])
            test_data.append({
                "tensor": torch.tensor(t),
                "pushes_raw": float(row["pushes"]),
                "pushes_norm": (np.log1p(float(row["pushes"])) - pushes_mean) / (pushes_std + 1e-8),
                "shell_hash": row["shell_hash"],
                "bucket": row["bucket"],
                "source": row["source"],
                "weight": 1.0,
            })

        val_data = []
        for _, row in val_df.iterrows():
            t = encode_board(row["board_str"])
            val_data.append({
                "tensor": torch.tensor(t),
                "pushes_raw": float(row["pushes"]),
                "pushes_norm": (np.log1p(float(row["pushes"])) - pushes_mean) / (pushes_std + 1e-8),
                "shell_hash": row["shell_hash"],
                "bucket": row["bucket"],
                "source": row["source"],
                "weight": 1.0,
            })

        train_data = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), leave=False, desc=f"Augmenting Fold {fold_num}"):
            t = encode_board(row["board_str"])
            for t_aug in augment_tensor(t):
                train_data.append({
                    "tensor": torch.from_numpy(t_aug.copy()),
                    "pushes_raw": float(row["pushes"]),
                    "pushes_norm": (np.log1p(float(row["pushes"])) - pushes_mean) / (pushes_std + 1e-8),
                    "shell_hash": row["shell_hash"],
                    "bucket": row["bucket"],
                    "source": row["source"],
                    "weight": float(class_weights.get(row["box_count"], 1.0)),
                })

        train_path = os.path.join(RESULTS_DIR, f"regressor_v2_fold{fold_num}_train.pt")
        val_path   = os.path.join(RESULTS_DIR, f"regressor_v2_fold{fold_num}_val.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_v2_fold{fold_num}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_v2_fold{fold_num}_stats.pt")

        torch.save(train_data, train_path)
        torch.save(val_data, val_path)
        torch.save(test_data, test_path)
        torch.save(stats, stats_path)

if __name__ == "__main__":
    main()
