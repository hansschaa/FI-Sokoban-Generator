import os
import subprocess
import tempfile
import sys
import pandas as pd
import numpy as np

# TODO: Rellena esta ruta con el archivo .pt que confirmemos en el lab
MODEL_PATH = "AQUI_VA_LA_RUTA"

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
    # 17 tableros de la intersección estricta (0 al 16)
    target_boards_ids = list(range(17))
    
    possible_files = ['sok_files/benchmark_stratified.sok', 'sok_files/paper.sok']
    boards = None
    for pf in possible_files:
        if os.path.exists(pf):
            boards = extract_boards(pf)
            if len(boards) > max(target_boards_ids):
                break
    
    if not boards:
        print("No se encontró el archivo .sok correcto con suficientes tableros.")
        sys.exit(1)
        
    print("Tableros extraídos correctamente.")
    
    # Crear archivo sok filtrado
    fd, temp_sok_filename = tempfile.mkstemp(prefix="phase1_", suffix=".sok")
    
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
    env['MODEL_PATH'] = MODEL_PATH # Enviamos el modelo por variable de entorno
    
    solver_bin = "./build/batch_solver"
    
    results = []
    
    for h in HEURISTICS:
        print(f"\n===========================================")
        print(f"  Heurística: {h}")
        print(f"===========================================")
        for rep in range(1, N_REPS + 1):
            temp_out_tsv = f"temp_out_phase1_{h}_{rep}.tsv"
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
                        nodes = int(parts[2]) if parts[2].isdigit() else -1
                        runtime = float(parts[3])
                        
                        internal_id = row_idx - 1
                        real_id = actual_board_map.get(internal_id, -1)
                        if real_id != -1:
                            results.append({
                                'board_id': real_id,
                                'heuristic': h,
                                'rep': rep,
                                'status': status,
                                'nodes': nodes,
                                'runtime_ms': runtime
                            })
                os.remove(temp_out_tsv)
                
    os.remove(temp_sok_filename)
    
    df = pd.DataFrame(results)
    csv_out = 'benchmark_phase1_results.csv'
    df.to_csv(csv_out, index=False)
    
    print(f"\n\n=== REPORTE MEDIANAS (GUARDADO EN {csv_out}) ===")
    
    for h in HEURISTICS:
        h_df = df[df['heuristic'] == h]
        times = []
        nodes_list = []
        solved = 0
        
        for bid in target_boards_ids:
            b_data = h_df[h_df['board_id'] == bid]
            if len(b_data) == 0:
                continue
                
            status = b_data['status'].iloc[0]
            if status == "SOLVED":
                solved += 1
                times.append(np.median(b_data['runtime_ms'].values))
                nodes_list.append(np.median(b_data['nodes'].values))
                
        if len(times) > 0:
            print(f"{h:25}: Solved {solved}/{len(target_boards_ids)} | Avg Nodes: {np.mean(nodes_list):.1f} | Avg Time: {np.mean(times):.1f} ms")
        else:
            print(f"{h:25}: Ningún resuelto.")

if __name__ == '__main__':
    main()
