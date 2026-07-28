import os
import subprocess
import csv
import time
import tempfile
import sys
import re
from collections import deque

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
            
    # Flush the last one if no trailing newline
    if current_board:
        boards.append((board_id, '\n'.join(current_board)))
        
    return boards

def preprocess_board(board_str):
    lines = board_str.split('\n')
    if not lines:
        return board_str
    
    max_len = max(len(line) for line in lines)
    # Rellenar con espacios para que sea rectangular
    grid = [list(line.ljust(max_len, ' ')) for line in lines]
    
    rows = len(grid)
    cols = max_len
    
    # Flood fill exterior spaces con '#'
    q = deque()
    
    for r in range(rows):
        if grid[r][0] == ' ': q.append((r, 0)); grid[r][0] = '#'
        if grid[r][cols-1] == ' ': q.append((r, cols-1)); grid[r][cols-1] = '#'
    for c in range(cols):
        if grid[0][c] == ' ': q.append((0, c)); grid[0][c] = '#'
        if grid[rows-1][c] == ' ': q.append((rows-1, c)); grid[rows-1][c] = '#'
        
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == ' ':
                grid[nr][nc] = '#'
                q.append((nr, nc))
                
    return '\n'.join(''.join(row) for row in grid)

def run_solver(board_str, heuristic, board_id, timeout_sec=180):
    """
    Ejecuta el test_solver con la heurística dada en el tablero provisto.
    Retorna un diccionario con los resultados, o maneja los errores/timeouts.
    """
    processed_board = preprocess_board(board_str)
    
    # 1. Crear un archivo temporal con el tablero
    fd, temp_filename = tempfile.mkstemp(prefix=f"sokoban_temp_{board_id}_", suffix=".txt")
    with os.fdopen(fd, 'w') as f:
        f.write(processed_board)
        
    # 2. Preparar el comando
    cmd = ['./build/test_solver', temp_filename, heuristic]
    
    # Aseguramos que la variable de entorno CUDA esté disponible
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    
    try:
        # 3. Ejecutar y capturar salida
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=env)
        output = result.stdout
        
        metrics = {
            'status': 'UNKNOWN',
            'runtime_ms': -1.0,
            'pushes': -1,
            'expanded_nodes': -1,
            'total_children': -1,
            'effective_children': -1,
            'deadlocks': -1
        }
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('status:'):
                metrics['status'] = line.split(':')[1].strip()
            elif line.startswith('runtime_ms:'):
                metrics['runtime_ms'] = float(line.split(':')[1].strip())
            elif line.startswith('pushes:'):
                metrics['pushes'] = int(line.split(':')[1].strip())
            elif line.startswith('expanded_nodes:'):
                metrics['expanded_nodes'] = int(line.split(':')[1].strip())
            elif line.startswith('total_children:'):
                metrics['total_children'] = int(line.split(':')[1].strip())
            elif line.startswith('effective_children:'):
                metrics['effective_children'] = int(line.split(':')[1].strip())
            elif line.startswith('deadlocks:'):
                metrics['deadlocks'] = int(line.split(':')[1].strip())
                
        return metrics
    except subprocess.TimeoutExpired:
        return {
            'status': 'HARD_TIMEOUT',
            'runtime_ms': 180000.0,
            'pushes': -1,
            'expanded_nodes': -1,
            'total_children': -1,
            'effective_children': -1,
            'deadlocks': -1
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

import argparse

def main():
    parser = argparse.ArgumentParser(description='Benchmark Sokoban Solver')
    parser.add_argument('--file', type=str, required=True, help='Ruta al archivo .sok')
    parser.add_argument('--start', type=int, default=0, help='ID del tablero inicial')
    parser.add_argument('--end', type=int, default=10, help='ID del tablero final (no inclusivo)')
    parser.add_argument('--resume', action='store_true', help='Reanudar desde el último progreso en el archivo CSV de salida')
    args = parser.parse_args()

    HEURISTICS = ['manhattan', 'hungarian', 'neural_sequential', 'neural_batched']
    
    csv_filename = f'benchmark_results_{args.start}_to_{args.end}.csv'
    
    # Manejar modo resume
    start_idx = args.start
    file_mode = 'w'
    if args.resume and os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            if len(rows) > 1: # Si tiene header y data
                last_board = int(rows[-1][0])
                start_idx = last_board + 1
        file_mode = 'a'
        print(f"[{csv_filename} encontrado] Reanudando desde el tablero {start_idx}...")

    sok_path = args.file
    print(f"Extrayendo tableros de {sok_path}...")
    boards = extract_boards(sok_path)
    
    # ── Dummy Warm-up para Inicializar CUDA ──
    print("\n[Warming Up] Realizando corridas dummy para inicializar pesos y CUDA si está disponible...")
    try:
        # Usa el primer tablero disponible para calentar (con timeout corto)
        warmup_board = boards[0][1]
        run_solver(warmup_board, 'neural_sequential', "warmup", timeout_sec=5)
        run_solver(warmup_board, 'neural_batched', "warmup", timeout_sec=5)
        print("[Warming Up] Listo.")
    except Exception as e:
        print(f"[Warming Up] Advertencia: falló el warmup ({e})")
    
    with open(csv_filename, file_mode, newline='') as csvfile:
        fieldnames = ['board_id', 'heuristic', 'status', 'runtime_ms', 'pushes', 
                      'expanded_nodes', 'total_children', 'effective_children', 'deadlocks']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if file_mode == 'w':
            writer.writeheader()
            
        print("\nINICIANDO BENCHMARK. Los resultados se guardarán en tiempo real en", csv_filename)
        print("-" * 60)
        
        for idx in range(start_idx, min(args.end, len(boards))):
            board_id, board_str = boards[idx]
            print(f"\nProcesando Tablero {board_id}...")
            
            for heuristic in HEURISTICS:
                print(f"  -> Ejecutando: {heuristic} (3 corridas)...")
                
                # Ejecutar 3 veces para reducir ruido del sistema (varianza)
                runtimes = []
                metrics_list = []
                last_metrics = None
                for _ in range(3):
                    metrics = run_solver(board_str, heuristic, board_id)
                    metrics_list.append(metrics)
                    if metrics['runtime_ms'] > 0:
                        runtimes.append(metrics['runtime_ms'])
                    last_metrics = metrics
                
                # Check determinism across the 3 runs (if they were solved/finished)
                if len(metrics_list) == 3 and metrics_list[0]['status'] != 'HARD_TIMEOUT':
                    for i in range(1, 3):
                        if (metrics_list[i]['pushes'] != metrics_list[0]['pushes'] or
                            metrics_list[i]['expanded_nodes'] != metrics_list[0]['expanded_nodes'] or
                            metrics_list[i]['deadlocks'] != metrics_list[0]['deadlocks']):
                            print(f"  [WARNING] Resultados no deterministas detectados en tablero {board_id} con {heuristic}!")
                            break

                # Promediar tiempos si las corridas fueron exitosas
                if runtimes and last_metrics['status'] == 'SOLVED':
                    last_metrics['runtime_ms'] = sum(runtimes) / len(runtimes)
                    
                row = {'board_id': board_id, 'heuristic': heuristic}
                row.update(last_metrics)
                
                writer.writerow(row)
                csvfile.flush() # Guardar inmediatamente en disco
                
                print(f"     [Status: {last_metrics['status']}, Time: {last_metrics['runtime_ms']:.2f}ms, Nodes: {last_metrics['expanded_nodes']}]")

if __name__ == "__main__":
    main()
