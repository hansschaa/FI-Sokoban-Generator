import os
import subprocess
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from pathlib import Path

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def generate_shells():
    print("Generando 600 cascarones (200x2x2, 200x2x3, 200x3x3)...")
    runner_path = "./build/test_shell"
    if not os.path.exists(runner_path):
        runner_path = "./build2/test_shell"
        
    shells = []
    for fx, fy, count in [(2, 2, 200), (2, 3, 200), (3, 3, 200)]:
        print(f"  Ejecutando {fx}x{fy} ({count})...")
        cmd = [runner_path, str(fx), str(fy), str(count)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        
        current_grid = []
        for line in res.stdout.splitlines():
            line = line.rstrip('\n').rstrip('\r')
            if line.startswith("====="):
                if current_grid:
                    shells.append(current_grid)
                current_grid = []
            elif line.startswith("Generando") or line.strip() == "":
                continue
            else:
                current_grid.append(line)
        if current_grid:
            shells.append(current_grid)
    return shells

def parse_sok_file(filepath):
    grid = []
    with open(filepath, 'r') as f:
        for line in f:
            l = line.rstrip('\n').rstrip('\r')
            if not l.startswith(';') and l.strip() != "":
                # Keep only walls and empty spaces (strip player/boxes if any, though shells shouldn't have them)
                clean_l = "".join(['#' if c == '#' else ' ' for c in l])
                grid.append(clean_l)
    return grid

# --- Métricas Estructurales ---
def normalize_grid(grid):
    w = max((len(r) for r in grid), default=0)
    return [r.ljust(w) for r in grid]

def find_bounding_box(grid):
    wall_positions = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == '#']
    if not wall_positions: return 0, max(0, len(grid)-1), 0, max(0, len(grid[0])-1)
    rs = [p[0] for p in wall_positions]
    cs = [p[1] for p in wall_positions]
    return min(rs), max(rs), min(cs), max(cs)

def get_interior_cells(grid):
    rows, cols = len(grid), len(grid[0])
    exterior = set()
    q = deque()
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r][c] == ' ' and (r,c) not in exterior:
                exterior.add((r,c)); q.append((r,c))
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r][c] == ' ' and (r,c) not in exterior:
                exterior.add((r,c)); q.append((r,c))
    while q:
        r,c = q.popleft()
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc = r+dr,c+dc
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in exterior and grid[nr][nc]==' ':
                exterior.add((nr,nc)); q.append((nr,nc))

    candidates = {(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==' ' and (r,c) not in exterior}
    visited = set()
    largest = set()
    regions = []
    for start in candidates:
        if start not in visited:
            region = set()
            q2 = deque([start])
            visited.add(start)
            while q2:
                r,c = q2.popleft()
                region.add((r,c))
                for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr,nc = r+dr,c+dc
                    if (nr,nc) in candidates and (nr,nc) not in visited:
                        visited.add((nr,nc)); q2.append((nr,nc))
            regions.append(region)
            if len(region) > len(largest):
                largest = region
    return largest, len(regions)

def get_inner_walls(grid, interior_cells):
    inner = set()
    rows, cols = len(grid), len(grid[0])
    for r, c in interior_cells:
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc] == '#':
                inner.add((nr, nc))
    return inner

def avg_symmetry(grid):
    min_r, max_r, min_c, max_c = find_bounding_box(grid)
    matches_h = total_h = 0
    for r in range(min_r, max_r+1):
        for c in range(min_c, max_c+1):
            mirror_c = max_c - (c - min_c)
            if mirror_c <= max_c:
                total_h += 1
                if grid[r][c] == grid[r][mirror_c]: matches_h += 1
    h_sym = matches_h / total_h if total_h > 0 else 0
    
    matches_v = total_v = 0
    for r in range(min_r, max_r+1):
        mirror_r = max_r - (r - min_r)
        for c in range(min_c, max_c+1):
            if mirror_r <= max_r:
                total_v += 1
                if grid[r][c] == grid[mirror_r][c]: matches_v += 1
    v_sym = matches_v / total_v if total_v > 0 else 0
    return (h_sym + v_sym) / 2

def compute_metrics(grid):
    grid = normalize_grid(grid)
    interior, num_regions = get_interior_cells(grid)
    inner_walls = get_inner_walls(grid, interior)
    denom_inner = len(interior) + len(inner_walls)
    
    min_r, max_r, min_c, max_c = find_bounding_box(grid)
    bb_total = (max_r-min_r+1) * (max_c-min_c+1)
    bb_walls = sum(1 for r in range(min_r, max_r+1) for c in range(min_c, max_c+1) if grid[r][c] == '#')
    
    dead_ends = sum(1 for r,c in interior if sum(1 for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)) if (r+dr,c+dc) in interior) == 1)
    
    # Nuevas directivas del plan: incluir free_cells explícitamente
    free_cells = len(interior)
    
    return {
        'wall_density': bb_walls / bb_total if bb_total > 0 else 0,
        'open_space_ratio': len(interior) / denom_inner if denom_inner > 0 else 0,
        'connectivity': len(interior),
        'aspect_ratio': (max_c-min_c+1) / (max_r-min_r+1) if (max_r-min_r+1) > 0 else 1,
        'dead_end_ratio': dead_ends / len(interior) if len(interior) > 0 else 0,
        'avg_symmetry': avg_symmetry(grid),
        'num_interior_regions': num_regions,
        'free_cells': free_cells  # CRÍTICO: Feature explícita
    }

def main():
    # 1. Generar pool
    raw_shells = generate_shells()
    print(f"Total generados: {len(raw_shells)}")
    
    records = []
    valid_shells = []
    for i, grid in enumerate(raw_shells):
        if not grid or not any('#' in r for r in grid): continue
        metrics = compute_metrics(grid)
        if metrics['free_cells'] < 9: continue # Ignorar triviales muy pequeños
        metrics['id'] = i
        records.append(metrics)
        valid_shells.append(grid)
        
    df = pd.DataFrame(records)
    print(f"Tableros válidos procesados: {len(df)}")
    
    # Features for clustering
    features = ['wall_density', 'open_space_ratio', 'connectivity', 'aspect_ratio', 
                'dead_end_ratio', 'avg_symmetry', 'num_interior_regions', 'free_cells']
    
    X = df[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Clustering K-Means K=5
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    sil_score = silhouette_score(X_scaled, df['cluster'])
    print(f"\n[K-Means K=5] Silhouette Score: {sil_score:.3f}")
    
    # Identificar el shell más cercano a cada centroide
    dist_to_centroids = cdist(X_scaled, centroids, metric='euclidean')
    closest_idxs = np.argmin(dist_to_centroids, axis=0)
    
    print("\n[Centroides Seleccionados]")
    new_centroid_grids = []
    for cluster_id, idx in enumerate(closest_idxs):
        original_id = int(df.iloc[idx]['id'])
        new_centroid_grids.append(valid_shells[original_id])
        print(f" Cluster {cluster_id} -> Shell de Pool #{original_id} (free_cells: {df.iloc[idx]['free_cells']})")
        # Guardar archivo .sok
        with open(f"levels/centroid_shell_{cluster_id+1}.sok", 'w') as f:
            for row in valid_shells[original_id]:
                f.write(row + "\n")
    print("Guardados como levels/centroid_shell_1.sok a centroid_shell_5.sok")
    
    # 3. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    centroids_pca = pca.transform(centroids)
    
    # 4. Leer y comparar shells actuales
    old_shell_files = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    old_records = []
    old_valid = []
    for sf in old_shell_files:
        if os.path.exists(sf):
            g = parse_sok_file(sf)
            if g:
                m = compute_metrics(g)
                m['name'] = os.path.basename(sf)
                old_records.append(m)
                old_valid.append(g)
    
    old_df = pd.DataFrame(old_records)
    if not old_df.empty:
        X_old = old_df[features].values
        X_old_scaled = scaler.transform(X_old)
        X_old_pca = pca.transform(X_old_scaled)
        old_df['pca1'] = X_old_pca[:, 0]
        old_df['pca2'] = X_old_pca[:, 1]
        
        # Criterio cuantitativo de reemplazo:
        # Calcular distancias de todos los shells del pool a su centroide más cercano
        min_dists = np.min(dist_to_centroids, axis=1)
        p75 = np.percentile(min_dists, 75)
        
        # Distancia de los old shells a CUALQUIER centroide nuevo
        old_dist_to_new = cdist(X_old_scaled, centroids, metric='euclidean')
        min_dists_old = np.min(old_dist_to_new, axis=1)
        
        outside_p75_count = sum(1 for d in min_dists_old if d > p75)
        
        print("\n[Evaluación de Criterio de Reemplazo]")
        print(f" Umbral Percentil 75 de distancias intra-cluster: {p75:.4f}")
        for i, row in old_df.iterrows():
            status = "FUERA" if min_dists_old[i] > p75 else "DENTRO"
            print(f" {row['name']:>12} -> Dist mínima al nuevo centroide: {min_dists_old[i]:.4f} ({status})")
            
        print(f"\n -> Shells actuales fuera del umbral p75: {outside_p75_count}/5")
        if outside_p75_count >= 3:
            print(" -> RESULTADO: Criterio CUMPLIDO. La selección actual es sustancialmente distinta al espacio procedural base y DEBE REEMPLAZARSE por los nuevos centroid_shell_X.sok")
        else:
            print(" -> RESULTADO: Criterio NO CUMPLIDO. La selección actual es suficientemente representativa del espacio estructural.")
            
    # Graficar
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', alpha=0.5, legend=False)
    
    # Marcar los centroides nuevos
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, marker='X', color='black', label='Nuevos Centroides')
    
    # Marcar los shells viejos
    if not old_df.empty:
        plt.scatter(old_df['pca1'], old_df['pca2'], s=200, marker='^', color='red', edgecolor='white', label='Shells Originales (1-5)')
        for i, row in old_df.iterrows():
            plt.annotate(row['name'].split('.')[0], (row['pca1'], row['pca2']), textcoords="offset points", xytext=(0,10), ha='center', color='darkred', weight='bold')

    plt.title("Espacio Estructural de Cascarones (PCA)\nNuevos Centroides vs Selección Original")
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    plt.ylabel(f"PCA 1 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scratch/shell_pca.pdf")
    print("\nGráfico PCA guardado en scratch/shell_pca.pdf")

if __name__ == "__main__":
    main()
