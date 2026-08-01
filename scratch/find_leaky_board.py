import sys
import pandas as pd
import subprocess
import os

sys.path.append('.')
from prepare_path_consistency import parse_sok_files, build_fold_map

def simulate_path_old(board_str, lurd_path):
    lines = [list(line) for line in board_str.splitlines()]
    max_w = max(len(l) for l in lines)
    for l in lines:
        l.extend(['#'] * (max_w - len(l)))
    
    px, py = -1, -1
    for r, row in enumerate(lines):
        for c, char in enumerate(row):
            if char in ['@', '+']:
                px, py = r, c
                break
        if px != -1: break
        
    dirs = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1),
            'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
            
    for m in lurd_path:
        if m not in dirs: continue
        dx, dy = dirs[m]
        nx, ny = px + dx, py + dy
        is_push = m.isupper()
        if is_push:
            bx, by = nx + dx, ny + dy
            box_char = lines[nx][ny]
            target_char = lines[bx][by]
            lines[nx][ny] = '-' if box_char == '$' else '.'
            lines[bx][by] = '$' if target_char in [' ', '-'] else '*'
        p_char = lines[px][py]
        lines[px][py] = ' ' if p_char == '@' else '.'
        n_char = lines[nx][ny]
        lines[nx][ny] = '@' if n_char in [' ', '-'] else '+'
        px, py = nx, ny

fold_map = build_fold_map()
records = parse_sok_files("../training_data/Solvables", fold_map)

sok_path = "temp_leak.sok"
tsv_path = "temp_leak.tsv"

for i, record in enumerate(records):
    board_str = record['board_str']
    
    with open(sok_path, "w") as f:
        f.write(f"Test - pushes:{record.get('pushes', 0)}\n")
        f.write(f"{board_str}\n\n")
        
    cmd = ["../build/batch_solver", sok_path, "hungarian", tsv_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
        df = pd.read_csv(tsv_path, sep='\t')
        if len(df) == 0: continue
        row = df.iloc[0]
        if row['Status'] != 'SOLVED' or row['LURD_Path'] == 'NONE': continue
        lurd = str(row['LURD_Path'])
    except Exception:
        continue
        
    try:
        simulate_path_old(board_str, lurd)
    except IndexError:
        print(f"\n=======================================================")
        print(f"TABLERO CORRUPTO ENCONTRADO EN EL DATASET:")
        print(f"Index: {i}")
        print(f"=======================================================\n")
        print(board_str)
        print(f"\n=======================================================")
        print(f"Ruta óptima devuelta por el motor en C++:")
        print(lurd)
        if os.path.exists(sok_path): os.remove(sok_path)
        if os.path.exists(tsv_path): os.remove(tsv_path)
        sys.exit(0)
        
    if i % 100 == 0:
        print(f"Processed {i} boards...")

if os.path.exists(sok_path): os.remove(sok_path)
if os.path.exists(tsv_path): os.remove(tsv_path)
