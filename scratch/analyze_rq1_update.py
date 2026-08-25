import pandas as pd
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import glob

def main():
    # 1. Load data
    csv_file = None
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Siempre tomar el más reciente por defecto
        files = glob.glob('benchmark_phase1_results_*.csv')
        if files:
            csv_file = max(files, key=os.path.getctime)
        else:
            csv_file = 'benchmark_results_0_to_40_new.csv'
            
    print(f"Leyendo archivo: {csv_file}")
    df_raw = pd.read_csv(csv_file)
    
    if 'expanded_nodes' in df_raw.columns:
        df_raw = df_raw.rename(columns={'expanded_nodes': 'nodes'})
    
    # Agregar las 5 repeticiones calculando la mediana de los tiempos y tomando el status
    df = df_raw.groupby(['board_id', 'heuristic']).agg({
        'status': lambda x: 'SOLVED' if 'SOLVED' in x.values else x.iloc[0],
        'nodes': 'median',
        'runtime_ms': 'median'
    }).reset_index()
    
    # 2. Filter heuristics
    core_heuristics = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched_massive']
    df = df[df['heuristic'].isin(core_heuristics)]
    
    # 3. Global Solved count
    print("=== Tableros Resueltos Globales (de 40) ===")
    solved_df = df[df['status'] == 'SOLVED']
    global_solved = solved_df.groupby('heuristic').size()
    for h in core_heuristics:
        count = global_solved.get(h, 0)
        print(f"{h:<25}: {count}/40 ({(count/40)*100:.1f}%)")
        
    # 4. Strict Intersection
    # Count how many heuristics solved each board
    board_solved_counts = solved_df.groupby('board_id').size()
    intersection_boards = board_solved_counts[board_solved_counts == len(core_heuristics)].index.tolist()
    
    print(f"\n=== Intersección Estricta (Resueltos por las 4) ===")
    print(f"Total: {len(intersection_boards)} tableros")
    print(f"Tableros: {intersection_boards}")
    
    # Median nodes/time in intersection
    intersect_df = solved_df[solved_df['board_id'].isin(intersection_boards)]
    stats = intersect_df.groupby('heuristic').agg({'nodes': 'median', 'runtime_ms': 'median'}).reindex(core_heuristics)
    print(f"\nMedianas en Intersección Estricta (n={len(intersection_boards)}):")
    print(stats)
    
    # 5. Wilcoxon Paired Test
    print("\n=== Test de Wilcoxon Pareado (Nodos Expandidos) ===")
    
    def paired_test(h1, h2):
        # We find boards where BOTH h1 and h2 solved
        h1_solved = solved_df[solved_df['heuristic'] == h1]['board_id']
        h2_solved = solved_df[solved_df['heuristic'] == h2]['board_id']
        common = list(set(h1_solved).intersection(set(h2_solved)))
        
        if not common:
            print(f"{h1} vs {h2}: No common solved boards.")
            return
            
        data_h1 = solved_df[(solved_df['heuristic'] == h1) & (solved_df['board_id'].isin(common))].sort_values('board_id')['nodes'].values
        data_h2 = solved_df[(solved_df['heuristic'] == h2) & (solved_df['board_id'].isin(common))].sort_values('board_id')['nodes'].values
        
        if len(common) > 0:
            try:
                # Wilcoxon signed-rank test
                res = wilcoxon(data_h1, data_h2)
                p_val = res.pvalue
            except Exception as e:
                p_val = float('nan')
                print(f"Wilcoxon error: {e}")
                
            # % of boards where h2 explores fewer nodes than h1
            h2_wins = sum(data_h2 < data_h1)
            h2_win_pct = (h2_wins / len(common)) * 100
            
            print(f"[{h1} vs {h2}] en {len(common)} tableros:")
            print(f"  - p-value: {p_val:.4e}")
            print(f"  - {h2} exploró menos nodos en {h2_wins}/{len(common)} tableros ({h2_win_pct:.1f}%)")
    
    paired_test('hungarian', 'neural_batched_massive')
    paired_test('hungarian', 'neural_sequential')
    
    # 6. Timeouts analysis
    print("\n=== Análisis de Timeouts ===")
    timeouts = df[df['status'] == 'TIMEOUT']
    timeouts_by_board = timeouts.groupby(['board_id', 'heuristic']).size().unstack(fill_value=0)
    print("Timeouts per board (only showing boards with timeouts):")
    print(timeouts_by_board)

if __name__ == '__main__':
    main()
