import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collections import deque
import re
import csv

def compute_distances(grid):
    goals = []
    boxes = []
    walls = set()
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    
    for r in range(rows):
        for c in range(len(grid[r])):
            ch = grid[r][c]
            if ch in ['.', '*', '+']:
                goals.append((r, c))
            if ch in ['$', '*']:
                boxes.append((r, c))
            if ch == '#':
                walls.add((r, c))
                
    dist_matrix = np.zeros((len(boxes), len(goals)))
    for g_idx, start in enumerate(goals):
        q = deque([(start[0], start[1], 0)])
        visited = {start}
        dists = {}
        while q:
            r, c, d = q.popleft()
            dists[(r, c)] = d
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in walls and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc, d+1))
        
        for b_idx, box in enumerate(boxes):
            dist_matrix[b_idx, g_idx] = dists.get(box, 9999)
            
    return dist_matrix, boxes, walls, rows, cols

def compute_features(grid_lines, pushes):
    if not grid_lines: return None
    dist_matrix, boxes, walls, rows, cols = compute_distances(grid_lines)
    
    if len(boxes) == 0: return None
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    hungarian_lb = dist_matrix[row_ind, col_ind].sum()
    
    pairwise_dists = []
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            d = abs(boxes[i][0] - boxes[j][0]) + abs(boxes[i][1] - boxes[j][1])
            pairwise_dists.append(d)
    
    dispersion = np.std(pairwise_dists) if pairwise_dists else 0.0
    
    area = rows * cols
    wall_density = len(walls) / area if area > 0 else 0
    
    gap = pushes - hungarian_lb
    
    return {
        'box_count': len(boxes),
        'gt_pushes': pushes,
        'hungarian_lb': hungarian_lb,
        'gap': gap,
        'dispersion': dispersion,
        'wall_density': wall_density,
        'area': area
    }

def parse_holdout_file(filepath):
    boards = {}
    current_grid = []
    board_id = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('; board_id='):
                if current_grid and board_id is not None:
                    boards[board_id] = current_grid
                    current_grid = []
                m = re.search(r'board_id=(\d+)', line)
                if m:
                    board_id = int(m.group(1))
            elif line.startswith('#'):
                current_grid.append(line)
            elif line == '' and current_grid and board_id is not None:
                boards[board_id] = current_grid
                current_grid = []
                board_id = None
                
    if current_grid and board_id is not None:
        boards[board_id] = current_grid
        
    return boards

def parse_train_file(filepath):
    boards = []
    current_grid = []
    pushes = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith(';'): continue
            if 'pushes:' in line:
                if current_grid:
                    boards.append((current_grid, pushes))
                    current_grid = []
                m = re.search(r'pushes:(\d+)', line)
                if m:
                    pushes = int(m.group(1))
            elif line.startswith('#'):
                current_grid.append(line)
                
    if current_grid:
        boards.append((current_grid, pushes))
        
    return boards

def main():
    failed_board_ids = {27, 31, 33, 34, 36}
    
    print("Cargando pushes reales desde benchmark_results_0_to_40.csv...")
    holdout_pushes_map = {}
    try:
        with open('benchmark_results_0_to_40.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['heuristic'] == 'hungarian' and row['status'] == 'SOLVED':
                    holdout_pushes_map[int(row['board_id'])] = int(row['pushes'])
    except Exception as e:
        print("Error leyendo CSV. Asegurate de tener benchmark_results_0_to_40.csv en este directorio.")
        return
        
    print("Cargando tableros holdout...")
    holdout_grids = parse_holdout_file('sok_files/benchmark_stratified_heldout.sok')
    
    holdout_features = {}
    for bid, grid in holdout_grids.items():
        pushes = holdout_pushes_map.get(bid, 0)
        if pushes > 0:
            feats = compute_features(grid, pushes)
            if feats:
                feats['board_id'] = bid
                holdout_features[bid] = feats
                
    train_features = []
    print("Extrayendo features del dataset de entrenamiento (buscando k-vecinos en muestra de ~9000 tableros)...")
    for bc in [5, 6, 7]:
        count = 0
        search_path = f"training_data/Solvables/{bc}/**/*.sok"
        for filepath in glob.glob(search_path, recursive=True):
            if count >= 3000: break
            boards = parse_train_file(filepath)
            for grid, pushes in boards:
                feats = compute_features(grid, pushes)
                if feats and feats['box_count'] == bc:
                    train_features.append(feats)
                    count += 1
                    if count >= 3000: break
                        
    train_df = pd.DataFrame(train_features)
    
    failed_data = []
    success_data = []
    
    for bid, feats in holdout_features.items():
        if bid in failed_board_ids:
            failed_data.append(feats)
        else:
            if feats['box_count'] in [5, 6, 7]:
                success_data.append(feats)
                
    failed_df = pd.DataFrame(failed_data)
    success_df = pd.DataFrame(success_data)
    
    print(f"\nDatos extraidos: {len(train_df)} en dataset de entrenamiento, {len(failed_df)} fallidos, {len(success_df)} exitosos (con 5-7 cajas).")
    
    feature_cols = ['dispersion', 'gt_pushes', 'gap', 'wall_density', 'area']
    
    for bc in [5, 6, 7]:
        print(f"\n" + "="*80)
        print(f"=== Analisis de Densidad Estructural para Box Count = {bc} ===")
        print("="*80)
        
        train_sub = train_df[train_df['box_count'] == bc]
        failed_sub = failed_df[failed_df['box_count'] == bc]
        success_sub = success_df[success_df['box_count'] == bc]
        
        if failed_sub.empty:
            print("No hay tableros fallidos con esta cantidad de cajas en este grupo.")
            continue
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_sub[feature_cols])
        
        k = 50
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_train)
        
        X_fail = scaler.transform(failed_sub[feature_cols])
        dists_fail, _ = knn.kneighbors(X_fail)
        avg_dist_fail = dists_fail.mean(axis=1)
        
        print(f"\n[!] Tableros Fallidos:")
        for idx, row in failed_sub.reset_index().iterrows():
            bid = int(row['board_id'])
            print(f"  Board {bid}: Dist Prom a {k}-NN = {avg_dist_fail[idx]:.4f} | Dispersion={row['dispersion']:5.2f} | Pushes={row['gt_pushes']:4.0f} | Gap={row['gap']:4.0f} | WallDens={row['wall_density']:.3f}")
            
        if not success_sub.empty:
            X_succ = scaler.transform(success_sub[feature_cols])
            dists_succ, _ = knn.kneighbors(X_succ)
            avg_dist_succ = dists_succ.mean(axis=1)
            print(f"\n[v] Tableros Exitosos (Muestra de {len(success_sub)} tableros holdout):")
            print(f"  Distancia Promedio a {k}-NN de TODOS los exitosos : {avg_dist_succ.mean():.4f}")
            print(f"  Dist Promedio Maxima en un tablero exitoso    : {avg_dist_succ.max():.4f}\n")
            
            for idx, row in success_sub.reset_index().iterrows():
                bid = int(row['board_id'])
                print(f"  Board {bid} (Exitoso): Dist Prom = {avg_dist_succ[idx]:.4f} | Dispersion={row['dispersion']:5.2f} | Pushes={row['gt_pushes']:4.0f} | Gap={row['gap']:4.0f} | WallDens={row['wall_density']:.3f}")
        else:
            print("No hay tableros exitosos en este box count en el holdout para comparar.")

if __name__ == '__main__':
    main()
