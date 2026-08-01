import torch
import numpy as np
import sys

path = "surrogate_models/results/path_consistency/path_fold3_train_part0.pt"
data = torch.load(path, map_location='cpu', weights_only=False)
pair = data[0]
s1 = np.array(pair['tensor1'])

# Encontrar límites reales
active_rows = np.where(s1.sum(axis=(0, 2)) > 0)[0]
active_cols = np.where(s1.sum(axis=(0, 1)) > 0)[0]
min_r, max_r = active_rows.min(), active_rows.max()
min_c, max_c = active_cols.min(), active_cols.max()

print("========================================")
print("1. TABLERO RECONSTRUIDO DESDE EL TENSOR")
print("========================================")
print("(Este es el tablero tal cual como la red lo armaría en su 'mente')")
for r in range(min_r, max_r + 1):
    row_str = ""
    for c in range(min_c, max_c + 1):
        is_wall = s1[1, r, c]
        is_box = s1[2, r, c]
        is_goal = s1[3, r, c]
        is_player = s1[4, r, c]
        
        if is_player and is_goal: row_str += "+"
        elif is_player: row_str += "@"
        elif is_box and is_goal: row_str += "*"
        elif is_box: row_str += "$"
        elif is_goal: row_str += "."
        elif is_wall: row_str += "#"
        else: row_str += " "
    print(row_str)

print("\n========================================")
print("2. CÓMO SE SEPARA EN CANALES MATEMÁTICOS")
print("========================================")

names = ["Canal 1: Paredes (#)", "Canal 2: Cajas ($)", "Canal 3: Metas (.)", "Canal 4: Jugador (@)"]
for ch_idx, name in zip([1, 2, 3, 4], names):
    print(f"\n{name}:")
    for r in range(min_r, max_r + 1):
        row_str = ""
        for c in range(min_c, max_c + 1):
            val = int(s1[ch_idx, r, c])
            row_str += "1 " if val else ". "
        print(row_str)

