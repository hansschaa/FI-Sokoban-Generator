import torch
import numpy as np
import sys
sys.path.append('surrogate_models')

path = "surrogate_models/results/path_consistency/path_fold3_train_part0.pt"
try:
    data = torch.load(path, map_location='cpu', weights_only=False)
except Exception as e:
    print(f"No se pudo cargar {path}: {e}")
    sys.exit(1)

print(f"Total de pares en este dataset de prueba: {len(data)}")
pair = data[0]

print("\n--- ESTADO 1 ---")
print(f"Empujes restantes: {pair['pushes1']}")
s1 = np.array(pair['tensor1'])
print("Tensor Shape:", s1.shape)

active_rows = np.where(s1.sum(axis=(0, 2)) > 0)[0]
active_cols = np.where(s1.sum(axis=(0, 1)) > 0)[0]
if len(active_rows) > 0 and len(active_cols) > 0:
    min_r, max_r = active_rows.min(), active_rows.max()
    min_c, max_c = active_cols.min(), active_cols.max()
else:
    min_r, max_r, min_c, max_c = 0, 19, 0, 19

channel_names = ["C0: Espacio Libre", "C1: Paredes (#)", "C2: Cajas ($)", "C3: Metas (.)", "C4: Jugador (@)", "C5: Deadlocks"]

for c in range(5):
    print(f"\n{channel_names[c]}:")
    for r in range(min_r, max_r + 1):
        row_str = ""
        for col in range(min_c, max_c + 1):
            val = int(s1[c, r, col])
            row_str += "1 " if val == 1 else ". "
        print(row_str)

print("\n--- ESTADO 2 (Después del empuje) ---")
print(f"Empujes restantes: {pair['pushes2']}")
s2 = np.array(pair['tensor2'])
print("C2: Cajas ($) EN ESTADO 2:")
for r in range(min_r, max_r + 1):
    row_str = ""
    for col in range(min_c, max_c + 1):
        val = int(s2[2, r, col])
        row_str += "1 " if val == 1 else ". "
    print(row_str)

