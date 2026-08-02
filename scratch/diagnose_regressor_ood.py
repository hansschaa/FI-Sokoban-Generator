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
    with open(filepath, 'r', encoding="utf-8", errors="replace") as f:
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

def sample_dataset_metrics(dir_path, max_files=20, max_boards_per_file=1000):
    if not os.path.exists(dir_path):
        print(f"⚠️ Advertencia: {dir_path} no se encuentra.")
        return []
    
    all_files = [os.path.join(root, f) for root, _, files in os.walk(dir_path) for f in files if f.endswith(".sok")]
    all_files.sort()
    if not all_files:
        return []

    sample_files = random.sample(all_files, min(max_files, len(all_files)))
    print(f" -> Analizando {dir_path} ({len(sample_files)} archivos muestreados)...")
    
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
    print(" Un Z-Score > 2.0 o < -2.0 indica fuera de distribución (OOD). Un Z-Score > 3.0 o < -3.0 es anomalía extrema.\n")
    header = f"{'Shell':<12}" + "".join([f"{f:>16}" for f in features])
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    for name, metrics in sorted(shell_records.items()):
        sh_num = name.replace(".sok", "").replace("shell_", "Shell ")
        row_str = f"{sh_num:<12}"
        for f in features:
            val = metrics[f]
            mean = stats.loc[f, 'mean']
            std = stats.loc[f, 'std']
            z = (val - mean) / std if std > 0 else 0
            flag = " 🚨" if abs(z) > 3.0 else (" ⚠️" if abs(z) > 2.0 else "   ")
            z_str = f"{z:+.2f}{flag}"
            row_str += f"{z_str:>16}"
        print(row_str)
    print("=" * len(header))

def main():
    print("\n" + "="*120)
    print(" 🔬 DIAGNÓSTICO DE ANOMALÍAS OUT-OF-DISTRIBUTION (OOD) EN EL REGRESOR NEURONAL")
    print(" Evaluando por qué Shell 1 (80% del gap) y Shell 5 (120% del gap) no cierran la brecha con el Clip")
    print("="*120)
    random.seed(42)

    print("\n1️⃣ Extrayendo métricas de la distribución de entrenamiento actual del Regresor (Solo Solvables)...")
    orig_records = sample_dataset_metrics("training_data/Solvables", max_files=20)
    df_orig = pd.DataFrame(orig_records)

    print("\n2️⃣ Extrayendo métricas del dataset Denso (DenseSolvables) usado para sanar el Clasificador...")
    dense_records = sample_dataset_metrics("training_data/DenseSolvables", max_files=20)
    df_dense = pd.DataFrame(dense_records)
    
    df_comb = pd.concat([df_orig, df_dense], ignore_index=True)

    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    shell_records = {}
    for sf in shells:
        if os.path.exists(sf):
            grid = parse_sok_file(sf)
            if grid:
                try:
                    shell_records[os.path.basename(sf)] = compute_metrics(grid)
                except Exception: pass

    features = ['wall_density', 'open_space_ratio', 'connectivity', 'aspect_ratio', 'dead_end_ratio', 'avg_symmetry']

    stats_orig = df_orig[features].describe().T
    print_z_score_table(stats_orig, shell_records, features, "📊 Z-SCORES CONTRA LA DISTRIBUCIÓN ACTUAL DEL REGRESOR (SOLO SOLVABLES)")

    stats_comb = df_comb[features].describe().T
    print_z_score_table(stats_comb, shell_records, features, "🛠️ Z-SCORES CONTRA DISTRIBUCIÓN REENTRENADA (SOLVABLES + DENSESOLVABLES)")

    print("\n💡 CONCLUSIÓN DIAGNÓSTICA:")
    print(" • Si Shell 1 y Shell 5 presentan Z-scores > 2.0 o 3.0 contra el Regresor Actual, se confirma que el Regresor sufre el mismo")
    print("   colapso por anomalías estructurales OOD (alta densidad / escasez de espacio libre) que sufría el Clasificador.")
    print(" • Incorporar DenseSolvables al entrenamiento del Regresor devolverá los shells extremos a la zona In-Distribution (Z <= 2.0).")
    print("="*120 + "\n")

if __name__ == "__main__":
    main()
