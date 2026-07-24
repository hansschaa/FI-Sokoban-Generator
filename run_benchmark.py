import os
import subprocess
import csv
import time

def extract_boards(sok_file):
    with open(sok_file, 'r') as f:
        content = f.read()
    
    boards = []
    current_board = []
    board_id = 0
    for line in content.split('\n'):
        if line.startswith('#') or (line.strip().startswith('#') and '#' in line):
            current_board.append(line)
        elif line.strip() == '' and current_board:
            boards.append((board_id, '\n'.join(current_board)))
            current_board = []
            board_id += 1
            
    # Flush the last one if no trailing newline
    if current_board:
        boards.append((board_id, '\n'.join(current_board)))
        
    return boards

def run_solver(board_str, heuristic):
    with open('temp_board.txt', 'w') as f:
        f.write(board_str)
    
    cmd = ['./build/test_solver', 'temp_board.txt', heuristic]
    try:
        # 180 seconds to allow the internal 120s C++ timeout to trigger gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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
    args = parser.parse_args()

    sok_path = 'sok_files/paper.sok'
    
    print(f"Extrayendo tableros de {sok_path}...")
    boards = extract_boards(sok_path)
    total_boards = len(boards)
    
    start_idx = max(0, args.start)
    end_idx = args.end if args.end > 0 else total_boards
    end_idx = min(end_idx, total_boards)
    
    boards = boards[start_idx:end_idx]
    print(f"Procesando chunk de tableros: {start_idx} a {end_idx} (Total: {len(boards)})")
    
    csv_path = f'benchmark_results_{start_idx}_to_{end_idx}.csv'
    heuristics = ['hungarian', 'neural', 'neural_batched']
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['board_id', 'heuristic', 'status', 'runtime_ms', 'pushes', 
                      'expanded_nodes', 'total_children', 'effective_children', 'deadlocks']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        print("\nINICIANDO BENCHMARK. Los resultados se guardarán en tiempo real en", csv_path)
        print("-" * 60)
        
        for board_id, board_str in boards:
            print(f"Procesando Tablero {board_id}...")
            
            for heuristic in heuristics:
                print(f"  -> Ejecutando: {heuristic}...")
                metrics = run_solver(board_str, heuristic)
                
                row = {'board_id': board_id, 'heuristic': heuristic}
                row.update(metrics)
                
                writer.writerow(row)
                csv_file.flush() # Guardar inmediatamente en disco
                
                print(f"     [Status: {metrics['status']}, Time: {metrics['runtime_ms']}ms, Nodes: {metrics['expanded_nodes']}]")

if __name__ == "__main__":
    main()
