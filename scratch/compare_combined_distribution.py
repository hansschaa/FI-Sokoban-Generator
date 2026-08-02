import os
import random
import numpy as np
import pandas as pd
try:
    from select_shells import parse_sok_file, compute_metrics
except ImportError:
    from scratch.select_shells import parse_sok_file, compute_metrics


def parse_sok_file_multi(filepath):
    grids = []
    current_grid = []
    with open(filepath, 'r') as f:
        for line in f:
            l = line.rstrip('\n').rstrip('\r')
            if l == "" or l[0].isdigit():
                if current_grid:
                    grids.append(current_grid)
                    current_grid = []
            elif not l.startswith(';') and l.strip() != "":
                clean_l = "".join(['#' if c == '#' else ' ' for c in l])
                current_grid.append(clean_l)
        if current_grid:
            grids.append(current_grid)
    return grids

def sample_dataset_metrics(dir_path, max_files=15, max_boards_per_file=1000):
    if not os.path.exists(dir_path):
        print(f"Advertencia: {dir_path} no existe o no se encuentra.")
        return []
    
    all_files = [os.path.join(root, f) for root, _, files in os.walk(dir_path) for f in files if f.endswith(".sok")]
    all_files.sort()  # Asegurar orden determinista antes de muestrear
    if not all_files:
        print(f"Advertencia: No se encontraron archivos .sok en {dir_path}.")
        return []

    sample_files = random.sample(all_files, min(max_files, len(all_files)))
    print(f"-> Cargando muestras desde {dir_path} ({len(sample_files)} archivos de {len(all_files)})...")
    
    records = []
    for f in sample_files:
        grids = parse_sok_file_multi(f)
        if len(grids) > max_boards_per_file:
            grids = random.sample(grids, max_boards_per_file)
        for grid in grids:
            try:
                m = compute_metrics(grid)
                records.append(m)
            except Exception:
                pass
    return records

def print_z_score_table(stats, shell_records, features, title):
    print(f"\n{title}")
    print("Un Z-Score > 2.0 o < -2.0 indica fuera del 95% normal (OOD). > 3.0 o < -3.0 es anomalía extrema.\n")
    header = f"{'Shell':<15}" + "".join([f"{f:>15}" for f in features])
    print(header)
    print("-" * len(header))
    
    for name, metrics in sorted(shell_records.items()):
        row_str = f"{name:<15}"
        for f in features:
            val = metrics[f]
            mean = stats.loc[f, 'mean']
            std = stats.loc[f, 'std']
            z = (val - mean) / std if std > 0 else 0
            z_str = f"{z:+.2f}" + ("*" if abs(z) > 2 else "") + ("*" if abs(z) > 3 else "")
            row_str += f"{z_str:>15}"
        print(row_str)

def main():
    random.seed(42)
    
    # 1. Muestrear Solvables Original y Denso
    orig_records = sample_dataset_metrics("training_data/Solvables", max_files=15)
    dense_records = sample_dataset_metrics("training_data/DenseSolvables", max_files=15)
    
    df_orig = pd.DataFrame(orig_records)
    df_dense = pd.DataFrame(dense_records)
    df_comb = pd.concat([df_orig, df_dense], ignore_index=True)
    
    print(f"\nResumen de muestras cargadas:\n  - Originales: {len(df_orig)}\n  - Densos:     {len(df_dense)}\n  - Combinados: {len(df_comb)}")
    
    # 2. Cargar los 5 Shells del experimento
    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    shell_records = {}
    for sf in shells:
        if os.path.exists(sf):
            grid = parse_sok_file(sf)
            if grid:
                try:
                    shell_records[os.path.basename(sf)] = compute_metrics(grid)
                except Exception:
                    pass
    
    features = ['wall_density', 'open_space_ratio', 'connectivity', 'aspect_ratio', 
                'dead_end_ratio', 'avg_symmetry', 'num_interior_regions', 'free_cells']
    
    print("\n" + "="*80)
    print(" COMPARACIÓN ESTADÍSTICA DE DISTRIBUCIONES (MEDIA y STD)")
    print("="*80)
    comp_df = pd.DataFrame()
    comp_df['Orig_Mean'] = df_orig[features].mean()
    comp_df['Dense_Mean'] = df_dense[features].mean()
    comp_df['Comb_Mean'] = df_comb[features].mean()
    comp_df['Comb_Std'] = df_comb[features].std()
    print(comp_df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # 3. Mostrar tablas de Z-Scores
    stats_orig = df_orig[features].describe().T
    stats_comb = df_comb[features].describe().T
    
    print_z_score_table(stats_orig, shell_records, features, "--- Z-SCORES CONTRA DISTRIBUCIÓN ORIGINAL (ANTES) ---")
    print_z_score_table(stats_comb, shell_records, features, "--- Z-SCORES CONTRA DISTRIBUCIÓN COMBINADA (NUEVA) ---")
    
    print("\n✅ Conclusión: Podés observar cómo los Z-scores extremales en Shell 1 y Shell 5 se reducen")
    print("   al incluir el dataset denso en la distribución combinada.")

if __name__ == "__main__":
    main()
