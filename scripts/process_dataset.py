import os
import glob
import pandas as pd
import numpy as np
import argparse
from collections import deque
from sklearn.model_selection import train_test_split

def fill_exterior_with_walls(board_str):
    """
    Realiza un Flood-Fill desde los bordes para reemplazar cualquier 
    espacio exterior (fuera de los muros) con muros '#'.
    """
    lines = board_str.strip().split('\n')
    if not lines:
        return np.array([])
    
    max_len = max(len(row) for row in lines)
    grid = np.array([list(row.ljust(max_len, ' ')) for row in lines])
    
    H, W = grid.shape
    q = deque()
    visited = set()
    
    # Agregar todos los bordes que no sean muros a la cola
    for r in range(H):
        if grid[r, 0] != '#':
            q.append((r, 0))
            visited.add((r, 0))
        if grid[r, W-1] != '#':
            q.append((r, W-1))
            visited.add((r, W-1))
    for c in range(W):
        if grid[0, c] != '#':
            q.append((0, c))
            visited.add((0, c))
        if grid[H-1, c] != '#':
            q.append((H-1, c))
            visited.add((H-1, c))
            
    while q:
        r, c = q.popleft()
        grid[r, c] = '#' # Convertir a muro
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if grid[nr, nc] != '#' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    
    return grid

def encode_board(grid):
    """
    Convierte el grid 2D de caracteres en un tensor 3D One-Hot de 5 canales:
    0: Muros (#)
    1: Piso / Interior (Todo lo que no sea muro)
    2: Cajas ($, *)
    3: Metas (., *, +)
    4: Jugador (@, +)
    Retorna un tensor de forma (5, H, W)
    """
    H, W = grid.shape
    tensor = np.zeros((5, H, W), dtype=np.float32)
    
    for r in range(H):
        for c in range(W):
            char = grid[r, c]
            if char == '#':
                tensor[0, r, c] = 1.0
            else:
                tensor[1, r, c] = 1.0 # Todo lo que no es muro es piso
                
                if char in ['$', '*']:
                    tensor[2, r, c] = 1.0
                if char in ['.', '*', '+']:
                    tensor[3, r, c] = 1.0
                if char in ['@', '+']:
                    tensor[4, r, c] = 1.0
                    
    return tensor

def process_datasets(input_dir, output_dir):
    print(f"Buscando archivos CSV en {input_dir}...")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print("No se encontraron archivos CSV.")
        return
        
    print(f"Se encontraron {len(csv_files)} archivos. Cargando...")
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total de registros cargados: {len(df)}")
    
    # 1. Balanceo de Clases
    # Mantendremos todos los positivos y hard_negatives, y samplearemos easy_negatives para igualar
    positives = df[df['is_solvable'] == 1]
    hard_negatives = df[df['dataset_type'] == 'hard_negative']
    easy_negatives = df[df['dataset_type'] == 'easy_negative']
    
    target_count = len(positives)
    print(f"Balanceando dataset... (Positivos: {target_count}, Hard Negatives: {len(hard_negatives)}, Easy Negatives: {len(easy_negatives)})")
    
    if len(easy_negatives) > target_count:
        easy_negatives = easy_negatives.sample(n=target_count, random_state=42)
        
    balanced_df = pd.concat([positives, hard_negatives, easy_negatives]).reset_index(drop=True)
    print(f"Registros despues del balanceo: {len(balanced_df)}")
    
    # 2. Data Leakage (Particion por shell_hash)
    unique_shells = balanced_df['shell_hash'].unique()
    print(f"Cascarones unicos encontrados: {len(unique_shells)}")
    
    # Dividimos los hashes: 70% Train, 15% Val, 15% Test
    train_hashes, temp_hashes = train_test_split(unique_shells, test_size=0.30, random_state=42)
    val_hashes, test_hashes = train_test_split(temp_hashes, test_size=0.50, random_state=42)
    
    print(f"Distribucion de cascarones - Train: {len(train_hashes)}, Val: {len(val_hashes)}, Test: {len(test_hashes)}")
    
    splits = {
        'train': balanced_df[balanced_df['shell_hash'].isin(train_hashes)],
        'val': balanced_df[balanced_df['shell_hash'].isin(val_hashes)],
        'test': balanced_df[balanced_df['shell_hash'].isin(test_hashes)]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_df in splits.items():
        print(f"Procesando split '{split_name}' con {len(split_df)} registros...")
        
        tensors = []
        labels = []
        
        for _, row in split_df.iterrows():
            grid = fill_exterior_with_walls(row['board_string'])
            tensor = encode_board(grid)
            tensors.append(tensor)
            labels.append(row['is_solvable'])
            
        # Como los tensores tienen tamanos variables, usamos un array de objetos o guardamos como diccionario en npz
        # np.savez admite listas de arrays de distinto tamano pasando cada array como argumento posicional
        # pero es mas practico pasarlos como un Object Array guardado en npy o npz
        
        out_path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez(out_path, 
                 x=np.array(tensors, dtype=object), 
                 y=np.array(labels, dtype=np.int32))
        
        print(f"Guardado {split_name} en {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa datasets de Sokoban para FCN.")
    parser.add_argument('--input_dir', type=str, default='build', help='Directorio con los CSVs')
    parser.add_argument('--output_dir', type=str, default='dataset_fcn', help='Directorio de salida para los .npz')
    
    args = parser.parse_args()
    process_datasets(args.input_dir, args.output_dir)
