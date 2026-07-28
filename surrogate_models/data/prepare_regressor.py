"""
prepare_regressor.py
--------------------
Parsea los tableros solubles, calcula el shell_hash, aplica GroupKFold(5)
estratificado por bucket de dificultad, codifica a tensores 5-canales y
guarda los 5 pares (train/test) en ../results/.

Anti-leakage garantizado:
- GroupKFold agrupa por shell_hash: nunca un shell aparece en train y test
  del mismo fold.
- Data Augmentation D4 (x8) se aplica SOLO al conjunto de Train de cada fold,
  DESPUES del split.
- Los estadisticos de normalizacion de targets se calculan SOLO sobre el train
  de cada fold y se aplican al test.
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVABLES_DIR = os.path.join(BASE_DIR, "..", "..", "training_data", "Solvables")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 5
MAX_H = 25
MAX_W = 25

# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DEL FORMATO .SOK
# ─────────────────────────────────────────────────────────────────────────────

def parse_sok_file(fpath):
    """
    Lee un archivo .sok y extrae registros con:
      board_str, pushes, branching_effective, bucket, shell_hash
    """
    records = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Los tableros están separados por líneas en blanco
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        header = lines[0]
        board_lines = lines[1:]

        # Extraer métricas de la cabecera
        pushes = None
        branching = None

        m_pushes = re.search(r"pushes:(\d+)", header)
        m_branch = re.search(r"branching_effective:([\d.]+)", header)

        if m_pushes:
            pushes = int(m_pushes.group(1))
        if m_branch:
            branching = float(m_branch.group(1))

        # Solo usamos tableros con pushes válidos para el regresor
        if pushes is None or pushes <= 0:
            continue

        board_str = "\n".join(board_lines)

        # Calcular shell_hash: quitamos todo excepto los muros '#'
        # Dejamos espacios donde había cajas/metas/jugador
        MOBILE_CHARS = str.maketrans("$.*@+", "     ")
        shell_str = board_str.translate(MOBILE_CHARS)
        shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()

        # Hash del tablero completo para deduplicación
        board_hash = hashlib.sha256(board_str.encode()).hexdigest()

        # Bucket de dificultad (para estratificación)
        if pushes <= 10:
            bucket = "1_to_10"
        elif pushes > 100:
            bucket = "101_plus"
        else:
            lower = ((pushes - 1) // 10) * 10 + 1
            upper = lower + 9
            bucket = f"{lower}_to_{upper}"

        # Extraer cantidad de cajas
        box_count = board_str.count('$') + board_str.count('*')

        records.append({
            "board_str": board_str,
            "pushes": pushes,
            "branching_effective": branching if branching is not None else 1.0,
            "bucket": bucket,
            "shell_hash": shell_hash,
            "board_hash": board_hash,
            "box_count": box_count
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: PARSEO → DEDUPLICACION → GROUPKFOLD → TENSORES → GUARDADO
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PREPARANDO DATASET REGRESOR (5-Fold GroupKFold + D4)")
    print("=" * 60)

    # 1. Parsear todos los archivos .sok de solubles
    print("\n[1/5] Parseando archivos .sok de Solvables...")
    all_records = []
    sok_files = glob.glob(
        os.path.join(SOLVABLES_DIR, "**", "*.sok"), recursive=True
    )
    print(f"      Archivos .sok encontrados: {len(sok_files)}")

    for fpath in tqdm(sok_files, desc="Parseando"):
        all_records.extend(parse_sok_file(fpath))

    df = pd.DataFrame(all_records)
    print(f"      Registros totales parseados: {len(df)}")

    # 2. Deduplicar por hash del tablero completo
    print("\n[2/5] Deduplicando por board_hash...")
    df = df.drop_duplicates(subset=["board_hash"]).reset_index(drop=True)
    print(f"      Tableros únicos tras deduplicación: {len(df)}")
    print(f"      Shells únicos (cascarones): {df['shell_hash'].nunique()}")
    print(f"\n      Distribución por bucket:")
    print(df["bucket"].value_counts().sort_index().to_string())

    # 3. Verificar shells con múltiples tableros (la clave del leakage)
    shell_counts = df.groupby("shell_hash").size()
    shells_with_multiple = (shell_counts > 1).sum()
    print(f"\n      Shells con >1 tablero (riesgo de leakage): {shells_with_multiple}")
    print(f"      Tableros máximos por shell: {shell_counts.max()}")

    print(f"\n[3/5] Aplicando GroupKFold(n_splits={N_FOLDS})...")
    print("      → Groups = shell_hash (mismo cascarón nunca cruza train/test/val)")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["shell_hash"].values

    fold_splits = list(gkf.split(df, groups=groups))

    # Verificar que no hay leakage de shells
    print("\n      Verificación anti-leakage:")
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        train_shells = set(df.iloc[train_idx]["shell_hash"])
        test_shells = set(df.iloc[test_idx]["shell_hash"])
        leak = train_shells & test_shells
        status = "✅ OK" if len(leak) == 0 else f"❌ LEAK ({len(leak)} shells compartidos)"
        print(f"      Fold {fold_idx + 1}: Train={len(train_idx)} | Test={len(test_idx)} | {status}")

    # 5. Codificar a tensores y guardar por fold
    print(f"\n[4/5] Codificando tableros a tensores (6 canales, {MAX_H}x{MAX_W})...")
    print("      Data Augmentation D4 (x8) aplicada SOLO al train de cada fold.")

    for fold_idx, (outer_train_idx, test_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        print(f"\n  ── FOLD {fold_num}/{N_FOLDS} ──────────────────────────────")

        outer_train_df = df.iloc[outer_train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx]

        # Inner Split: separar 20% de outer_train_df para Validation
        inner_gkf = GroupKFold(n_splits=5)
        inner_groups = outer_train_df["shell_hash"].values
        # Tomamos el primer split interno como (train_final, val)
        inner_train_idx, val_idx = next(inner_gkf.split(outer_train_df, groups=inner_groups))
        
        train_df = outer_train_df.iloc[inner_train_idx]
        val_df = outer_train_df.iloc[val_idx]

        # Normalización estricta sobre el conjunto de Train (anti-leakage)
        pushes_train = train_df["pushes"].values.astype(float)
        pushes_log = np.log1p(pushes_train)
        pushes_mean = pushes_log.mean()
        pushes_std = pushes_log.std()

        pushes_mean = pushes_log.mean()
        pushes_std = pushes_log.std()

        # Calcular pesos de clase basados en box_count en el train
        box_counts = train_df["box_count"].value_counts()
        total_train = len(train_df)
        # weight = total_train / (num_classes * count), balancea los gradientes
        class_weights = {k: total_train / (len(box_counts) * v) for k, v in box_counts.items()}

        stats = {
            "pushes_mean": float(pushes_mean),
            "pushes_std": float(pushes_std),
        }
        print(f"      Stats (solo train): pushes={pushes_mean:.1f}±{pushes_std:.1f}")

        # Codificar TEST (sin augmentation, con normalización del train)
        print(f"      Codificando test ({len(test_df)} tableros)...")
        test_data = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
            t = encode_board(row["board_str"])
            test_data.append({
                "tensor": torch.tensor(t),
                "pushes_raw": float(row["pushes"]),
                "pushes_norm": (np.log1p(float(row["pushes"])) - pushes_mean) / (pushes_std + 1e-8),
                "shell_hash": row["shell_hash"],
                "bucket": row["bucket"],
                "weight": 1.0, # Test samples no afectan el loss
            })

        # Codificar VAL (sin augmentation, con normalización del train)
        print(f"      Codificando val ({len(val_df)} tableros)...")
        val_data = []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), leave=False):
            t = encode_board(row["board_str"])
            val_data.append({
                "tensor": torch.tensor(t),
                "pushes_raw": float(row["pushes"]),
                "pushes_norm": (np.log1p(float(row["pushes"])) - pushes_mean) / (pushes_std + 1e-8),
                "shell_hash": row["shell_hash"],
                "bucket": row["bucket"],
                "weight": 1.0,
            })

        # Codificar TRAIN (con augmentation D4 x8)
        print(f"      Codificando train ({len(train_df)} tableros + D4 x8 = ~{len(train_df)*8})...")
        train_data = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), leave=False):
            t = encode_board(row["board_str"])
            for t_aug in augment_tensor(t):
                train_data.append({
                    "tensor": torch.from_numpy(t_aug.copy()),
                    "pushes_raw": float(row["pushes"]),
                    "pushes_norm": (np.log1p(float(row["pushes"])) - pushes_mean) / (pushes_std + 1e-8),
                    "shell_hash": row["shell_hash"],
                    "bucket": row["bucket"],
                    "weight": float(class_weights.get(row["box_count"], 1.0)),
                })

        # Guardar
        train_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_train.pt")
        val_path   = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_val.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_stats.pt")

        torch.save(train_data, train_path)
        torch.save(val_data, val_path)
        torch.save(test_data, test_path)
        torch.save(stats, stats_path)

        print(f"      ✅ Guardado: fold{fold_num} train={len(train_data)} | val={len(val_data)} | test={len(test_data)}")

    print("\n[5/5] ✅ Proceso completado exitosamente.")
    print(f"      Archivos guardados en: {RESULTS_DIR}")
    print(f"      Archivos generados:")
    for f_num in range(1, N_FOLDS + 1):
        print(f"        regressor_fold{f_num}_train.pt | regressor_fold{f_num}_val.pt | regressor_fold{f_num}_test.pt | regressor_fold{f_num}_stats.pt")


if __name__ == "__main__":
    main()
