import sys
import numpy as np
from prepare_path_consistency import simulate_path
sys.path.append('../data')
from board_utils import encode_board

board_str = """#######
#     #
# $ . #
#   ###
#   
###"""

print("\n=======================================================")
print("  TEST VISUAL DE SANIDAD (1 Tablero de Muestra)")
print("=======================================================")
print("\n--- TABLERO ORIGINAL ---")
for i, line in enumerate(board_str.splitlines()):
    print(f"'{line}' (len={len(line)})")

states = simulate_path(board_str, "R")
s1_str = states[0][0]

print("\n--- TABLERO CON PADDING DE MUROS (#) APLICADO ---")
for line in s1_str.splitlines():
    print(f"'{line}'")

print("\n--- CÓMO LA RED NEURONAL LO VE EN CANALES ---")
t1 = encode_board(s1_str)

active_rows = np.where(t1.sum(axis=(0, 2)) > 0)[0]
active_cols = np.where(t1.sum(axis=(0, 1)) > 0)[0]
min_r, max_r = active_rows.min(), active_rows.max()
min_c, max_c = active_cols.min(), active_cols.max()

names = ["C0: Espacio Libre", "C1: Paredes (#)", "C2: Cajas ($)", "C3: Metas (.)", "C4: Jugador (@)"]
for ch_idx in range(5):
    print(f"\n{names[ch_idx]}:")
    for r in range(min_r, max_r + 1):
        row_str = ""
        for c in range(min_c, max_c + 1):
            val = int(t1[ch_idx, r, c])
            row_str += "1 " if val else ". "
        print(row_str)
print("=======================================================\n")
