import os
import subprocess
import csv
import time
import tempfile
import sys
import re

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

import argparse

def main():
    parser = argparse.ArgumentParser(description='Benchmark Sokoban Solver (Batch Mode)')
    parser.add_argument('--file', type=str, required=True, help='Ruta al archivo .sok')
    parser.add_argument('--start', type=int, default=0, help='ID del tablero inicial')
    parser.add_argument('--end', type=int, default=10, help='ID del tablero final (no inclusivo)')
    parser.add_argument('--resume', action='store_true', help='Ignorado en modo batch')
    args = parser.parse_args()

    HEURISTICS = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched', 'neural_batched_massive']
    csv_filename = f'benchmark_results_{args.start}_to_{args.end}.csv'
    
    boards = extract_boards(args.file)
    target_boards = boards[args.start:min(args.end, len(boards))]
    
    print(f"Extrayendo {len(target_boards)} tableros de {args.file} (desde {args.start} hasta {args.start+len(target_boards)-1})...")
    
    # 1. Crear un archivo temporal con los tableros seleccionados
    fd, temp_sok_filename = tempfile.mkstemp(prefix="sokoban_batch_temp_", suffix=".sok")
    with os.fdopen(fd, 'w') as f:
        for board_id, board_str in target_boards:
            f.write(board_str + "\n\n")

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['PYTORCH_JIT_USE_NVFuser'] = '0'

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['board_id', 'heuristic', 'status', 'runtime_ms', 'pushes', 
                      'expanded_nodes', 'total_children', 'effective_children', 'deadlocks']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print("\nINICIANDO BENCHMARK (Batch-Process por Heuristica)")
        print("-" * 60)
        
        for heuristic in HEURISTICS:
            print(f"\n===========================================")
            print(f"  Ejecutando: {heuristic}")
            print(f"===========================================")
            
            temp_out_tsv = f"temp_out_{heuristic}.tsv"
            cmd = ['./build/batch_solver', temp_sok_filename, heuristic, temp_out_tsv]
            
            try:
                # Damos 10 minutos para que complete todos los tableros de una heurística
                subprocess.run(cmd, env=env, check=True, timeout=600)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT EXPIRED for {heuristic}!")
            except subprocess.CalledProcessError as e:
                print(f"Error ejecutando {heuristic}: {e}")
                
            # Parse output TSV
            if os.path.exists(temp_out_tsv):
                with open(temp_out_tsv, 'r') as f:
                    for row_idx, line in enumerate(f):
                        if row_idx == 0: continue # Skip header
                        parts = line.strip().split('\t')
                        if len(parts) < 8: continue
                        
                        lvl_name = parts[0]
                        status = parts[1]
                        runtime = float(parts[3])
                        pushes = int(parts[4])
                        expanded = int(parts[6]) # generated_states en batch_solver corresponde a expanded en run_benchmark
                        deadlocks = int(parts[7])
                        
                        # Map back to original board id
                        board_id = args.start + row_idx - 1
                        
                        row = {
                            'board_id': board_id, 
                            'heuristic': heuristic, 
                            'status': status, 
                            'runtime_ms': runtime, 
                            'pushes': pushes, 
                            'expanded_nodes': expanded, 
                            'total_children': -1, 
                            'effective_children': -1, 
                            'deadlocks': deadlocks
                        }
                        writer.writerow(row)
                        csvfile.flush()
                        
                        print(f"  -> Tablero {board_id}: [Status: {status}, Time: {runtime:.2f}ms, Nodes: {expanded}]")
                os.remove(temp_out_tsv)

    os.remove(temp_sok_filename)
    print("\nBENCHMARK COMPLETADO.")

if __name__ == "__main__":
    main()
