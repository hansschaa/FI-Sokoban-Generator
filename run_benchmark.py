import os
import subprocess
import csv
import time
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

def run_solver(board_str, heuristic):
    processed_board = preprocess_board(board_str)
    with open('temp_board.txt', 'w') as f:
        f.write(processed_board)
    
    cmd = ['./build/test_solver', 'temp_board.txt', heuristic]
    try:
        # Prevent PyTorch from spawning hundreds of threads
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        env['OPENBLAS_NUM_THREADS'] = '1'
        
        # 180 seconds to allow the internal 120s C++ timeout to trigger gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
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

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Sokoban benchmarks.')
    parser.add_argument('--start', type=int, default=0, help='Start board index (inclusive)')
    parser.add_argument('--end', type=int, default=-1, help='End board index (exclusive)')
    parser.add_argument('--file', type=str, default='sok_files/auto_generated.sok', help='Path to .sok file')
    args = parser.parse_args()

    sok_path = args.file
    
    print(f"Extrayendo tableros de {sok_path}...")
    boards = extract_boards(sok_path)
    total_boards = len(boards)
    
    start_idx = max(0, args.start)
    end_idx = args.end if args.end > 0 else total_boards
    end_idx = min(end_idx, total_boards)
    
    boards = boards[start_idx:end_idx]
    print(f"Procesando chunk de tableros: {start_idx} a {end_idx} (Total: {len(boards)})")
    
    csv_path = f'benchmark_results_{start_idx}_to_{end_idx}.csv'
    heuristics = ['hungarian', 'neural_sequential', 'neural_batched']
    
    # ── Reanudación Segura (Resume) ──
    completed_runs = set()
    file_exists = os.path.isfile(csv_path)
    if file_exists:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_runs.add((int(row['board_id']), row['heuristic']))
                
    # ── Dummy Warm-up para Inicializar CUDA ──
    print("\n[Warming Up] Realizando corridas dummy en GPU para inicializar CUDA/pesos...")
    try:
        # Usa el primer tablero disponible para calentar
        warmup_board = boards[0][1]
        run_solver(warmup_board, 'neural_sequential')
        run_solver(warmup_board, 'neural_batched')
        print("[Warming Up] Listo.")
    except Exception as e:
        print(f"[Warming Up] Advertencia: falló el warmup ({e})")
    
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['board_id', 'heuristic', 'status', 'runtime_ms', 'pushes', 
                      'expanded_nodes', 'total_children', 'effective_children', 'deadlocks']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        print("\nINICIANDO BENCHMARK. Los resultados se guardarán en tiempo real en", csv_path)
        print("-" * 60)
        
        for board_id, board_str in boards:
            print(f"\nProcesando Tablero {board_id}...")
            
            for heuristic in heuristics:
                if (board_id, heuristic) in completed_runs:
                    print(f"  -> Saltando: {heuristic} (Ya procesado)")
                    continue
                    
                print(f"  -> Ejecutando: {heuristic} (3 corridas)...")
                
                # Ejecutar 3 veces para reducir ruido del sistema (varianza)
                runtimes = []
                last_metrics = None
                for _ in range(3):
                    metrics = run_solver(board_str, heuristic)
                    if metrics['runtime_ms'] > 0:
                        runtimes.append(metrics['runtime_ms'])
                    last_metrics = metrics
                
                # Promediar tiempos si las corridas fueron exitosas
                if runtimes and last_metrics['status'] == 'SOLVED':
                    last_metrics['runtime_ms'] = sum(runtimes) / len(runtimes)
                    
                row = {'board_id': board_id, 'heuristic': heuristic}
                row.update(last_metrics)
                
                writer.writerow(row)
                csv_file.flush() # Guardar inmediatamente en disco
                
                print(f"     [Status: {last_metrics['status']}, Time: {last_metrics['runtime_ms']:.2f}ms, Nodes: {last_metrics['expanded_nodes']}]")

if __name__ == "__main__":
    main()
