import os
import subprocess
import tempfile
import sys
import pandas as pd
import numpy as np

def extract_boards(sok_file):
    with open(sok_file, 'r') as f:
        content = f.read()
    
    boards = []
    current_board = []
    board_id = 0
    for line in content.split('\n'):
        if line.strip().startswith('#'):
            current_board.append(line)
        elif line.strip() == '' and current_board:
            boards.append((board_id, '\n'.join(current_board)))
            current_board = []
            board_id += 1
            
    if current_board:
        boards.append((board_id, '\n'.join(current_board)))
        
    return boards

def main():
    # Los 17 tableros de la intersección estricta original (0 al 16)
    target_boards_ids = list(range(17))
    
    # Intenta encontrar el archivo .sok correcto
    possible_files = ['sok_files/benchmark_stratified_heldout.sok', 'sok_files/benchmark_stratified.sok', 'sok_files/paper.sok']
    boards = None
    for pf in possible_files:
        if os.path.exists(pf):
            boards = extract_boards(pf)
            if len(boards) > max(target_boards_ids):
                break
    
    if not boards:
        print("No se encontró el archivo .sok correcto con suficientes tableros.")
        sys.exit(1)
        
    print(f"Tableros extraídos correctamente.")
    
    # Crear archivo sok filtrado
    fd, temp_sok_filename = tempfile.mkstemp(prefix="sokoban_verify_", suffix=".sok")
    
    # We want to maintain original board IDs in batch_solver output?
    # batch_solver simply outputs row index. We will keep a map.
    actual_board_map = {}
    with os.fdopen(fd, 'w') as f:
        idx = 0
        for bid, bstr in boards:
            if bid in target_boards_ids:
                f.write(f"Board_{bid}\n{bstr}\n\n")
                actual_board_map[idx] = bid
                idx += 1
                
    HEURISTICS = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched_massive']
    N_REPS = 5
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['PYTORCH_JIT_USE_NVFuser'] = '0'
    
    solver_bin = "./build/batch_solver"
    if not os.path.exists(solver_bin):
        solver_bin = "./build2/batch_solver"
        
    results = []
    
    for h in HEURISTICS:
        print(f"\n===========================================")
        print(f"  Heurística: {h}")
        print(f"===========================================")
        for rep in range(1, N_REPS + 1):
            temp_out_tsv = f"temp_out_{h}_{rep}.tsv"
            cmd = [solver_bin, temp_sok_filename, h, temp_out_tsv]
            print(f"  --> Repetición {rep}...")
            
            try:
                subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error en {h} rep {rep}: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                
            if os.path.exists(temp_out_tsv):
                with open(temp_out_tsv, 'r') as f:
                    for row_idx, line in enumerate(f):
                        if row_idx == 0: continue # header
                        parts = line.strip().split('\t')
                        if len(parts) < 8: continue
                        status = parts[1]
                        runtime = float(parts[3])
                        
                        internal_id = row_idx - 1
                        real_id = actual_board_map.get(internal_id, -1)
                        if real_id != -1:
                            results.append({
                                'board_id': real_id,
                                'heuristic': h,
                                'rep': rep,
                                'status': status,
                                'runtime_ms': runtime
                            })
                os.remove(temp_out_tsv)
                
    os.remove(temp_sok_filename)
    
    # Análisis
    df = pd.DataFrame(results)
    df.to_csv('scratch/verification_results.csv', index=False)
    
    print("\n\n=== REPORTE DE VARIANZA (N=5) ===")
    
    for h in HEURISTICS:
        print(f"\n--- {h} ---")
        h_df = df[df['heuristic'] == h]
        
        report_data = []
        for bid in target_boards_ids:
            b_data = h_df[h_df['board_id'] == bid]
            if len(b_data) == 0:
                continue
            
            times = b_data['runtime_ms'].values
            median = np.median(times)
            min_t = np.min(times)
            max_t = np.max(times)
            rng = max_t - min_t
            disp = (rng / median) * 100 if median > 0 else 0
            
            report_data.append({
                'board_id': bid,
                'median_ms': f"{median:.2f}",
                'range_ms': f"[{min_t:.2f} - {max_t:.2f}]",
                'dispersion_%': f"{disp:.2f}%",
                'disp_val': disp
            })
            
        report_df = pd.DataFrame(report_data)
        if not report_df.empty:
            print(report_df[['board_id', 'median_ms', 'range_ms', 'dispersion_%']].to_string(index=False))

if __name__ == '__main__':
    main()
