import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from tqdm import tqdm

def get_hungarian_lb(tensor_board):
    """
    Calcula el Hungarian Lower Bound para un tablero (6, 25, 25).
    0: muros, 1: metas, 2: cajas
    """
    walls = tensor_board[0].numpy() == 1
    goals = np.argwhere(tensor_board[1].numpy() == 1)
    boxes = np.argwhere(tensor_board[2].numpy() == 1)

    if len(goals) != len(boxes) or len(goals) == 0:
        return 0.0

    H, W = walls.shape
    
    # Pre-calcular BFS desde cada goal hacia todas las celdas alcanzables
    dist_maps = []
    for gr, gc in goals:
        dist = np.full((H, W), np.inf)
        dist[gr, gc] = 0
        q = deque([(gr, gc)])
        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not walls[nr, nc] and np.isinf(dist[nr, nc]):
                        dist[nr, nc] = d + 1
                        q.append((nr, nc))
        dist_maps.append(dist)

    # Construir matriz de costos Goal -> Box
    num_items = len(goals)
    cost_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j, (br, bc) in enumerate(boxes):
            cost_matrix[i, j] = dist_maps[i][br, bc]

    # Resolver asignación
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    lb = cost_matrix[row_ind, col_ind].sum()
    
    if np.isinf(lb):
        return 0.0 # Ocurre si hay cajas inalcanzables (deadlocks obvios)
    return float(lb)

def process_file(filepath):
    print(f"Procesando {filepath}...")
    data = torch.load(filepath, weights_only=False)
    for i in tqdm(range(len(data)), desc="Calculando Hungarian LB"):
        if 'hungarian_lb' not in data[i]:
            data[i]['hungarian_lb'] = get_hungarian_lb(data[i]['tensor'])
            
            # También pre-calculamos el residuo real (pushes - lb)
            # Nota: a veces el lb = 0 por bugs o deadlocks en el tensor, protegemos contra negativos absurdos
            residual = max(0, data[i]['pushes_raw'] - data[i]['hungarian_lb'])
            data[i]['residual_raw'] = residual
            
    torch.save(data, filepath)
    print(f"✅ Guardado: {filepath}")

def main():
    results_dir = "surrogate_models/results"
    
    # Procesar todo
    for fold in range(1, 6):
        for split in ['train', 'val', 'test']:
            f = os.path.join(results_dir, f"regressor_fold{fold}_{split}.pt")
            if os.path.exists(f):
                process_file(f)
                
    # Además calcular las estadísticas del residuo por fold
    for fold in range(1, 6):
        train_path = os.path.join(results_dir, f"regressor_fold{fold}_train.pt")
        stats_path = os.path.join(results_dir, f"regressor_fold{fold}_stats.pt")
        if os.path.exists(train_path) and os.path.exists(stats_path):
            data = torch.load(train_path, weights_only=False)
            stats = torch.load(stats_path, weights_only=False)
            
            residuals = [np.log1p(d['residual_raw']) for d in data]
            stats['residual_mean'] = float(np.mean(residuals))
            stats['residual_std'] = float(np.std(residuals))
            
            # Guardamos el norm en los datos
            for i in range(len(data)):
                data[i]['residual_norm'] = (np.log1p(data[i]['residual_raw']) - stats['residual_mean']) / stats['residual_std']
                
            torch.save(data, train_path)
            torch.save(stats, stats_path)
            
            # Y aplicar la normalización al val/test
            for split in ['val', 'test']:
                split_path = os.path.join(results_dir, f"regressor_fold{fold}_{split}.pt")
                if os.path.exists(split_path):
                    split_data = torch.load(split_path, weights_only=False)
                    for i in range(len(split_data)):
                        split_data[i]['residual_norm'] = (np.log1p(split_data[i]['residual_raw']) - stats['residual_mean']) / stats['residual_std']
                    torch.save(split_data, split_path)
                    
            print(f"✅ Fold {fold} stats actualizadas (Residual Mean: {stats['residual_mean']:.3f}, Std: {stats['residual_std']:.3f})")

if __name__ == "__main__":
    main()
