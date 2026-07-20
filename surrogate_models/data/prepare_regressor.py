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
from sklearn.model_selection import StratifiedGroupKFold

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

        records.append({
            "board_hash": board_hash,
            "shell_hash": shell_hash,
            "board_str": board_str,
            "pushes": pushes,
            "branching_effective": branching if branching is not None else 1.0,
            "bucket": bucket,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# CODIFICACION A TENSOR 5-CANALES
# ─────────────────────────────────────────────────────────────────────────────

def encode_board(board_str, max_h=MAX_H, max_w=MAX_W):
    """
    Convierte un string de tablero a un tensor float32 de forma (5, max_h, max_w).

    Canales:
      0 → Muros (#)
      1 → Interior caminable (flood-fill)
      2 → Cajas ($ o *)
      3 → Metas (. o * o +)
      4 → Jugador (@ o +)

    El padding exterior rellena el Canal 0 con muros (valor 1.0) en lugar de
    ceros, para que la CNN sepa que "afuera todo es roca".
    """
    lines = board_str.splitlines()
    H = len(lines)
    W = max(len(l) for l in lines) if lines else 0

    # Recortar si excede el tamaño máximo (no debería suceder con datos limpios)
    H = min(H, max_h)
    W = min(W, max_w)

    char_matrix = np.full((H, W), ' ', dtype=str)
    for r, line in enumerate(lines[:H]):
        for c, ch in enumerate(line[:W]):
            char_matrix[r, c] = ch

    # Flood-fill exterior para calcular zona interior caminable
    exterior = _flood_fill_exterior(char_matrix, H, W)

    # Tensor base: relleno con muros (canal 0 = 1.0) en toda la zona de padding
    tensor = np.zeros((5, max_h, max_w), dtype=np.float32)

    # La zona de padding (fuera del tablero real) son muros
    tensor[0, :, :] = 1.0

    # Centrar el tablero dentro del tensor max_h x max_w
    pad_top = (max_h - H) // 2
    pad_left = (max_w - W) // 2

    for r in range(H):
        for c in range(W):
            ch = char_matrix[r, c]
            tr = r + pad_top
            tc = c + pad_left

            if ch == '#':
                tensor[0, tr, tc] = 1.0
                continue

            # Zona interior: quitar el muro de fondo del padding
            tensor[0, tr, tc] = 0.0

            if not exterior[r, c]:
                tensor[1, tr, tc] = 1.0  # Interior caminable

            if ch in ('$', '*'):
                tensor[2, tr, tc] = 1.0  # Caja

            if ch in ('.', '*', '+'):
                tensor[3, tr, tc] = 1.0  # Meta

            if ch in ('@', '+'):
                tensor[4, tr, tc] = 1.0  # Jugador

    return tensor


def _flood_fill_exterior(char_matrix, H, W):
    """BFS desde los bordes para marcar el exterior (fuera de las paredes)."""
    exterior = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)
    q = []

    for r in range(H):
        q.append((r, 0))
        q.append((r, W - 1))
    for c in range(W):
        q.append((0, c))
        q.append((H - 1, c))

    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        if r < 0 or r >= H or c < 0 or c >= W:
            continue
        if visited[r, c]:
            continue
        visited[r, c] = True
        if char_matrix[r, c] == '#':
            continue
        exterior[r, c] = True
        q.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    return exterior


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION D4 (8 SIMETRIAS)
# ─────────────────────────────────────────────────────────────────────────────

def augment_d4(tensor):
    """
    Genera las 8 variantes del grupo diédrico D4 (4 rotaciones x 2 reflejos).
    Retorna una lista de 8 tensores numpy (5, H, W).
    """
    augmented = []
    for k in range(4):
        rot = np.rot90(tensor, k=k, axes=(1, 2)).copy()
        augmented.append(rot)
        refl = np.flip(rot, axis=2).copy()
        augmented.append(refl)
    return augmented


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

    # 4. GroupKFold estratificado
    print(f"\n[3/5] Aplicando StratifiedGroupKFold(n_splits={N_FOLDS})...")
    print("      → Groups = shell_hash (mismo cascarón nunca cruza train/test)")
    print("      → Stratify = bucket (distribución de dificultad equilibrada)")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    groups = df["shell_hash"].values
    stratify_labels = df["bucket"].values

    fold_splits = list(sgkf.split(df, stratify_labels, groups=groups))

    # Verificar que no hay leakage de shells
    print("\n      Verificación anti-leakage:")
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        train_shells = set(df.iloc[train_idx]["shell_hash"])
        test_shells = set(df.iloc[test_idx]["shell_hash"])
        leak = train_shells & test_shells
        status = "✅ OK" if len(leak) == 0 else f"❌ LEAK ({len(leak)} shells compartidos)"
        print(f"      Fold {fold_idx + 1}: Train={len(train_idx)} | Test={len(test_idx)} | {status}")

    # 5. Codificar a tensores y guardar por fold
    print(f"\n[4/5] Codificando tableros a tensores (5 canales, {MAX_H}x{MAX_W})...")
    print("      Data Augmentation D4 (x8) aplicada SOLO al train de cada fold.")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        print(f"\n  ── FOLD {fold_num}/{N_FOLDS} ──────────────────────────────")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Calcular estadísticos de normalización SOLO sobre el train
        pushes_mean = train_df["pushes"].mean()
        pushes_std = train_df["pushes"].std()
        branch_mean = train_df["branching_effective"].mean()
        branch_std = train_df["branching_effective"].std()

        stats = {
            "pushes_mean": float(pushes_mean),
            "pushes_std": float(pushes_std),
            "branch_mean": float(branch_mean),
            "branch_std": float(branch_std),
        }
        print(f"      Stats (solo train): pushes={pushes_mean:.1f}±{pushes_std:.1f} | "
              f"branch={branch_mean:.2f}±{branch_std:.2f}")

        # Codificar TEST (sin augmentation, con normalización del train)
        print(f"      Codificando test ({len(test_df)} tableros)...")
        test_data = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
            t = encode_board(row["board_str"])
            test_data.append({
                "tensor": torch.tensor(t),
                "pushes_raw": float(row["pushes"]),
                "pushes_norm": (float(row["pushes"]) - pushes_mean) / (pushes_std + 1e-8),
                "branch_raw": float(row["branching_effective"]),
                "branch_norm": (float(row["branching_effective"]) - branch_mean) / (branch_std + 1e-8),
                "shell_hash": row["shell_hash"],
                "bucket": row["bucket"],
            })

        # Codificar TRAIN (con augmentation D4 x8)
        print(f"      Codificando train ({len(train_df)} tableros + D4 x8 = ~{len(train_df)*8})...")
        train_data = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), leave=False):
            t = encode_board(row["board_str"])
            variants = augment_d4(t)
            for v in variants:
                train_data.append({
                    "tensor": torch.tensor(v.copy()),
                    "pushes_raw": float(row["pushes"]),
                    "pushes_norm": (float(row["pushes"]) - pushes_mean) / (pushes_std + 1e-8),
                    "branch_raw": float(row["branching_effective"]),
                    "branch_norm": (float(row["branching_effective"]) - branch_mean) / (branch_std + 1e-8),
                    "shell_hash": row["shell_hash"],
                    "bucket": row["bucket"],
                })

        # Guardar
        train_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_train.pt")
        test_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold_num}_stats.pt")

        torch.save(train_data, train_path)
        torch.save(test_data, test_path)
        torch.save(stats, stats_path)

        print(f"      ✅ Guardado: fold{fold_num} train={len(train_data)} | test={len(test_data)}")

    print("\n[5/5] ✅ Proceso completado exitosamente.")
    print(f"      Archivos guardados en: {RESULTS_DIR}")
    print(f"      Archivos generados:")
    for f_num in range(1, N_FOLDS + 1):
        print(f"        regressor_fold{f_num}_train.pt | regressor_fold{f_num}_test.pt | regressor_fold{f_num}_stats.pt")


if __name__ == "__main__":
    main()
