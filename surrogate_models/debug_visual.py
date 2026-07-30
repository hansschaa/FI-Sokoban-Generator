import sys
import subprocess
import os
import numpy as np
import pandas as pd
from prepare_path_consistency import simulate_path, parse_sok_files, build_fold_map

sys.path.append('../data')
from board_utils import encode_board

print("\n=======================================================")
print("  TEST VISUAL DE SANIDAD (1er Tablero Real del Dataset)")
print("=======================================================")

fold_map = build_fold_map()
records = parse_sok_files("../training_data/Solvables", fold_map)
record = records[0]
board_str = record['board_str']

print("\n--- 1. TABLERO ORIGINAL (Desde el dataset) ---")
for i, line in enumerate(board_str.splitlines()):
    print(f"'{line}'")

sok_path = "temp_debug.sok"
tsv_path = "temp_debug.tsv"
with open(sok_path, "w") as f:
    f.write(f"Test - pushes:{record.get('pushes', 0)}\n")
    f.write(f"{board_str}\n\n")

print("\n--- 2. OBTENIENDO RUTA ÓPTIMA CON BATCH_SOLVER (C++) ---")
cmd = ["../build/batch_solver", sok_path, "hungarian", tsv_path]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

df = pd.read_csv(tsv_path, sep='\t')
lurd = str(df.iloc[0]['LURD_Path'])
print(f"Ruta óptima encontrada: {lurd[:50]}... (len={len(lurd)})")

print("\n--- 3. SIMULANDO EMPUJES Y APLICANDO PADDING DE MUROS ---")
states = simulate_path(board_str, lurd)
s1_str, pushes_left = states[0]

for line in s1_str.splitlines():
    print(f"'{line}'")

print("\n--- 4. TRADUCCIÓN MATEMÁTICA (TENSORES DE LA RED) ---")
print(f"Etiqueta de Entrenamiento (Empujes Restantes): {pushes_left}")
t1 = encode_board(s1_str)

active_rows = np.where(t1.sum(axis=(0, 2)) > 0)[0]
active_cols = np.where(t1.sum(axis=(0, 1)) > 0)[0]
min_r, max_r = active_rows.min(), active_rows.max()
min_c, max_c = active_cols.min(), active_cols.max()

names = ["C0: Paredes (#)", "C1: Espacio Libre", "C2: Cajas ($)", "C3: Metas (.)", "C4: Jugador (@)"]
for ch_idx in range(5):
    print(f"\n{names[ch_idx]}:")
    for r in range(min_r, max_r + 1):
        row_str = ""
        for c in range(min_c, max_c + 1):
            val = int(t1[ch_idx, r, c])
            row_str += "1 " if val else ". "
        print(row_str)
print("=======================================================\n")

os.remove(sok_path)
os.remove(tsv_path)
