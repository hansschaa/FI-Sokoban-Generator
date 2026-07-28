"""
run_benchmark_stratified.py
───────────────────────────
Runner del benchmark estratificado por dificultad.
Lee benchmark_stratified_meta.csv para saber el bucket de cada tablero
y aplica heurísticas y repeticiones distintas según la dificultad:

  Bucket 1-30   → todas las heurísticas (incl. neural_sequential), 3 repeticiones
  Bucket 31-70  → sin neural_sequential (timeouts frecuentes), 3 repeticiones
  Bucket 71+    → solo hungarian + neural_batched_massive, 2 repeticiones

Uso:
  python3 run_benchmark_stratified.py [--meta benchmark_stratified_meta.csv]
                                      [--sok  sok_files/benchmark_stratified.sok]
                                      [--resume]
"""

import os
import csv
import tempfile
import subprocess
import argparse
from collections import deque
import sys

# ── Heurísticas por bucket ────────────────────────────────────────────────────
BUCKET_CONFIG = {
    "1-10":   {"heuristics": ["manhattan", "hungarian", "neural_sequential", "neural_batched", "neural_batched_massive"], "reps": 1},
    "11-20":  {"heuristics": ["manhattan", "hungarian", "neural_sequential", "neural_batched", "neural_batched_massive"], "reps": 1},
    "21-30":  {"heuristics": ["manhattan", "hungarian", "neural_sequential", "neural_batched", "neural_batched_massive"], "reps": 1},
    "31-50":  {"heuristics": ["manhattan", "hungarian",                       "neural_batched", "neural_batched_massive"], "reps": 1},
    "51-70":  {"heuristics": ["manhattan", "hungarian",                       "neural_batched", "neural_batched_massive"], "reps": 1},
    "71-90":  {"heuristics": [             "hungarian",                                        "neural_batched_massive"], "reps": 1},
    "91-100": {"heuristics": [             "hungarian",                                        "neural_batched_massive"], "reps": 1},
    "101+":   {"heuristics": [             "hungarian",                                        "neural_batched_massive"], "reps": 1},
}


# ── Reutilizado de run_benchmark.py ──────────────────────────────────────────
def extract_boards(sok_file):
    """
    Extrae tableros del archivo .sok generado por build_difficulty_benchmark.py.
    Las líneas que empiezan con ';' son comentarios/metadata y se ignoran.
    """
    with open(sok_file, 'r') as f:
        content = f.read()
    boards = []
    current_board = []
    board_id = 0
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith(';'):
            continue  # línea de metadata, ignorar
        if stripped.startswith('#'):
            current_board.append(line)
        elif current_board:
            boards.append((board_id, '\n'.join(current_board)))
            current_board = []
            board_id += 1
    if current_board:
        boards.append((board_id, '\n'.join(current_board)))
    return boards


def preprocess_board(board_str):
    lines = board_str.split('\n')
    if not lines:
        return board_str
    max_len = max(len(l) for l in lines)
    grid = [list(l.ljust(max_len, ' ')) for l in lines]
    rows, cols = len(grid), max_len
    q = deque()
    for r in range(rows):
        if grid[r][0] == ' ':     q.append((r, 0));      grid[r][0] = '#'
        if grid[r][cols-1] == ' ':q.append((r, cols-1)); grid[r][cols-1] = '#'
    for c in range(cols):
        if grid[0][c] == ' ':      q.append((0, c));      grid[0][c] = '#'
        if grid[rows-1][c] == ' ': q.append((rows-1, c)); grid[rows-1][c] = '#'
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == ' ':
                grid[nr][nc] = '#'
                q.append((nr, nc))
    return '\n'.join(''.join(row) for row in grid)


def run_solver(board_str, heuristic, board_id, timeout_sec=180):
    processed_board = preprocess_board(board_str)
    fd, temp_filename = tempfile.mkstemp(prefix=f"sokoban_strat_{board_id}_", suffix=".txt")
    with os.fdopen(fd, 'w') as f:
        f.write(processed_board)
    cmd = ['./build/test_solver', temp_filename, heuristic]
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['PYTORCH_JIT_USE_NVFuser'] = '0'
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, env=env)
        output = result.stdout
        metrics = {
            'status': 'UNKNOWN', 'runtime_ms': -1.0, 'pushes': -1,
            'expanded_nodes': -1, 'total_children': -1,
            'effective_children': -1, 'deadlocks': -1,
        }
        for line in output.split('\n'):
            line = line.strip()
            if   line.startswith('status:'):           metrics['status']           = line.split(':',1)[1].strip()
            elif line.startswith('runtime_ms:'):       metrics['runtime_ms']       = float(line.split(':',1)[1].strip())
            elif line.startswith('pushes:'):           metrics['pushes']           = int(line.split(':',1)[1].strip())
            elif line.startswith('expanded_nodes:'):   metrics['expanded_nodes']   = int(line.split(':',1)[1].strip())
            elif line.startswith('total_children:'):   metrics['total_children']   = int(line.split(':',1)[1].strip())
            elif line.startswith('effective_children:'):metrics['effective_children']= int(line.split(':',1)[1].strip())
            elif line.startswith('deadlocks:'):        metrics['deadlocks']        = int(line.split(':',1)[1].strip())
        return metrics
    except subprocess.TimeoutExpired:
        return {'status': 'HARD_TIMEOUT', 'runtime_ms': 180000.0, 'pushes': -1,
                'expanded_nodes': -1, 'total_children': -1, 'effective_children': -1, 'deadlocks': -1}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def load_meta(meta_csv):
    """Devuelve dict: board_id → {bucket, box_count, source_file, board_hash}"""
    meta = {}
    with open(meta_csv, newline='') as f:
        for row in csv.DictReader(f):
            meta[int(row['board_id'])] = {
                'bucket':     row['bucket'],
                'box_count':  int(row['box_count']),
                'source_file':row['source_file'],
                'board_hash': row['board_hash'],
            }
    return meta


def main():
    parser = argparse.ArgumentParser(description='Benchmark estratificado por dificultad (Held-Out)')
    parser.add_argument('--meta', default='benchmark_stratified_heldout_meta.csv',
                        help='CSV de metadata generado por build_fold_heldout_benchmark.py')
    parser.add_argument('--sok',  default='sok_files/benchmark_stratified_heldout.sok',
                        help='Archivo .sok de tableros estratificados held-out')
    parser.add_argument('--out',  default='benchmark_heldout_results.csv',
                        help='Archivo CSV de resultados de salida')
    parser.add_argument('--resume', action='store_true',
                        help='Reanudar desde el último progreso')
    args = parser.parse_args()

    # ── Cargar metadata ───────────────────────────────────────────────────────
    meta = load_meta(args.meta)

    # ── Cargar tableros ───────────────────────────────────────────────────────
    boards = dict(extract_boards(args.sok))   # board_id → board_str
    print(f"Tableros cargados: {len(boards)}  |  Metadata: {len(meta)} entradas")

    # ── Resume ────────────────────────────────────────────────────────────────
    completed_triples = set()   # (board_id, heuristic, rep)
    file_mode = 'w'
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline='') as f:
            for row in csv.DictReader(f):
                completed_triples.add((int(row['board_id']), row['heuristic'], int(row['rep'])))
        file_mode = 'a'
        print(f"[Resume] {len(completed_triples)} tripletas ya completadas.")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("\n[Warmup] Inicializando CUDA/JIT...")
    wb = boards[0]
    run_solver(wb, 'neural_batched_massive', 'warmup', timeout_sec=30)
    run_solver(wb, 'hungarian', 'warmup', timeout_sec=30)
    print("[Warmup] Listo.\n")

    # ── CSV de salida ─────────────────────────────────────────────────────────
    fieldnames = [
        'board_id', 'bucket', 'box_count', 'source_file',
        'heuristic', 'rep',
        'status', 'runtime_ms', 'pushes', 'expanded_nodes',
        'total_children', 'effective_children', 'deadlocks',
    ]

    with open(args.out, file_mode, newline='') as csvout:
        writer = csv.DictWriter(csvout, fieldnames=fieldnames)
        if file_mode == 'w':
            writer.writeheader()

        for board_id in sorted(boards.keys()):
            if board_id not in meta:
                print(f"  [WARN] board_id={board_id} no tiene metadata, saltando.")
                continue

            bm      = meta[board_id]
            bucket  = bm['bucket']
            cfg     = BUCKET_CONFIG.get(bucket, BUCKET_CONFIG["101+"])
            heurs   = cfg['heuristics']
            reps    = cfg['reps']

            for heuristic in heurs:
                for rep in range(1, reps + 1):
                    key = (board_id, heuristic, rep)
                    if key in completed_triples:
                        print(f"  [skip] board={board_id}  {heuristic}  rep={rep}")
                        continue

                    # Timeout más largo para buckets difíciles
                    timeout = 200 if bucket in ("71-90", "91-100", "101+") else 150
                    print(f"  board={board_id:3d}  bucket={bucket:7s}  boxes={bm['box_count']}  "
                          f"{heuristic:24s}  rep={rep}/{reps} ...", end='', flush=True)

                    metrics = run_solver(boards[board_id], heuristic, board_id, timeout)
                    print(f"  {metrics['status']}  {metrics['runtime_ms']:.0f}ms  nodes={metrics['expanded_nodes']}")

                    row = {
                        'board_id':          board_id,
                        'bucket':            bucket,
                        'box_count':         bm['box_count'],
                        'source_file':       bm['source_file'],
                        'heuristic':         heuristic,
                        'rep':               rep,
                        **metrics,
                    }
                    writer.writerow(row)
                    csvout.flush()

    print(f"\n→ Resultados en: {args.out}")


if __name__ == '__main__':
    main()
