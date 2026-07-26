import torch
import numpy as np
import random
import os

RESULTS_DIR = "/home/hanss/FI-sokoban-generator/surrogate_models/results"

def decode_tensor(t):
    # t is shape (5, 25, 25)
    # Channels: 0: Wall, 1: Walkable, 2: Box, 3: Goal, 4: Player
    H, W = t.shape[1], t.shape[2]
    board = []
    for r in range(H):
        row = ""
        for c in range(W):
            if t[0, r, c] > 0.5:
                row += "#"
            elif t[2, r, c] > 0.5 and t[3, r, c] > 0.5:
                row += "*"
            elif t[2, r, c] > 0.5:
                row += "$"
            elif t[4, r, c] > 0.5 and t[3, r, c] > 0.5:
                row += "+"
            elif t[4, r, c] > 0.5:
                row += "@"
            elif t[3, r, c] > 0.5:
                row += "."
            elif t[1, r, c] > 0.5:
                row += " "
            else:
                row += " " # exterior padding or empty
        # Right strip to make it look clean like original board
        board.append(row.rstrip())
    
    # Remove empty rows at bottom/top
    while board and not board[-1]: board.pop()
    while board and not board[0]: board.pop(0)
    return "\n".join(board)

def main():
    print("Cargando dataset...")
    train_data = torch.load(f"{RESULTS_DIR}/regressor_fold1_train.pt", weights_only=False)
    stats = torch.load(f"{RESULTS_DIR}/regressor_fold1_stats.pt", weights_only=False)
    
    p_mean = stats["pushes_mean"]
    p_std = stats["pushes_std"]
    
    # Filtrar solo tableros con pocos empujes (ej. 1 a 15) para demostrar que se incluyeron
    low_pushes_data = [d for d in train_data if 1 <= d["pushes_raw"] <= 15]
    print(f"Total de tableros en Train: {len(train_data)}")
    print(f"Total de tableros de pocos empujes (1-15): {len(low_pushes_data)}")
    
    if len(low_pushes_data) == 0:
        print("¡ALERTA! No se encontraron tableros de pocos empujes.")
        return
        
    print("\n" + "="*50)
    print("MUESTRA DE 5 TABLEROS AL AZAR (POCOS EMPUJES)")
    print("="*50)
    
    samples = random.sample(low_pushes_data, 5)
    
    for i, s in enumerate(samples):
        print(f"\n--- MUESTRA {i+1} ---")
        print(f"Pushes Raw: {s['pushes_raw']}")
        p_desnorm = (s['pushes_norm'] * p_std) + p_mean
        p_desnorm_real = np.expm1(p_desnorm)
        print(f"Pushes desnormalizado (expm1): {p_desnorm_real:.2f} (Verificando integridad matemática)")
        print(f"Branching Effective (raw): {s['branch_raw']:.2f}")
        print(f"Bucket asignado: {s['bucket']}")
        print(f"Hash del shell: {s['shell_hash']}")
        print("Tensor reconstruido a texto:")
        print(decode_tensor(s['tensor'].numpy()))

if __name__ == "__main__":
    main()
