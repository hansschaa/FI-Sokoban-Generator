import os
import sys
import argparse
import json
import glob
import re
import hashlib
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import subprocess

# Importar la función canónica de encode_board desde board_utils
# NUNCA reimplementar esta función — ya causó bugs graves dos veces (C++ y aquí).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
from board_utils import encode_board

K_STEPS = 4
MAX_SAMPLES = 500
BUCKET_DIR = "../training_data/Solvables"
OUTPUT_DIR = "results/path_consistency"
OUTPUT_SOK = "../scratch/path_consistency_sample.sok"
OUTPUT_TSV = "../scratch/path_consistency_results.tsv"

def get_bucket(pushes):
    if pushes <= 10: return "1_to_10"
    if pushes > 100: return "101_plus"
    lower = ((pushes - 1) // 10) * 10 + 1
    upper = lower + 9
    return f"{lower}_to_{upper}"

def parse_sok_files(directory, fold_map):
    records = []
    files = glob.glob(os.path.join(directory, "**/*.sok"), recursive=True)
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3: continue
            header = lines[0]
            board_lines = lines[1:]
            
            m = re.match(r"(\d+)\s*-\s*(.*)", header)
            if not m: continue
            level_name = str(m.group(1)).strip()
            stats_str = m.group(2)
            
            pushes = -1
            if "pushes:" in stats_str:
                for token in stats_str.split():
                    if token.startswith("pushes:"):
                        pushes = int(token.split(":")[1])
            elif stats_str.isdigit():
                pushes = int(stats_str)
            
            board_str = "\n".join(board_lines)
            MOBILE_CHARS = str.maketrans("$.*@+", "     ")
            shell_str = board_str.translate(MOBILE_CHARS)
            shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()
            
            if shell_hash not in fold_map:
                continue
                
            records.append({
                "hash": shell_hash,
                "name": level_name,
                "board_str": board_str,
                "pushes": pushes,
                "bucket": get_bucket(pushes)
            })
        # The inner loop iterates over blocks. No max check here.
    return records

def simulate_path(board_str, lurd_path):
    lines = [list(l) for l in board_str.splitlines()]
    if lines:
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
        
    states = []
    total_pushes = sum(1 for m in lurd_path if m.isupper())
    
    # State zero
    states.append(("\n".join("".join(row) for row in lines), total_pushes))
    
    dirs = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1),
            'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    
    accumulated_pushes = 0
    
    for m in lurd_path:
        if m not in dirs: continue
        dx, dy = dirs[m]
        nx, ny = px + dx, py + dy
        
        is_push = m.isupper()
        if is_push:
            bx, by = nx + dx, ny + dy
            # Move box
            box_char = lines[nx][ny]
            target_char = lines[bx][by]
            
            lines[nx][ny] = '-' if box_char == '$' else '.'
            lines[bx][by] = '$' if target_char in [' ', '-'] else '*'
            accumulated_pushes += 1
            
        # Move player
        p_char = lines[px][py]
        lines[px][py] = ' ' if p_char == '@' else '.'
        n_char = lines[nx][ny]
        lines[nx][ny] = '@' if n_char in [' ', '-'] else '+'
        
        px, py = nx, ny
        
        # Take snapshot every K pushes
        if is_push and accumulated_pushes % K_STEPS == 0:
            states.append(("\n".join("".join(row) for row in lines), total_pushes - accumulated_pushes))
            
    # Always include the goal state if not already included
    if accumulated_pushes % K_STEPS != 0:
        states.append(("\n".join("".join(row) for row in lines), 0))
            
    return states

# NOTA: encode_board() se importa desde data/board_utils.py al inicio del archivo.
# No reimplementar aquí — la convención de canales es:
#   C0: Muros | C1: Interior/Piso | C2: Cajas | C3: Metas | C4: Jugador | C5: Deadlock mask

def build_fold_map():
    fold_map = {}
    fpath = "results/fold_map.json"
    print(f"Loading fold mapping from {fpath}...")
    if not os.path.exists(fpath):
        print(f"ERROR: {fpath} not found! Cannot build fold_map.")
        return fold_map
    
    with open(fpath, "r") as f:
        fold_map = json.load(f)
        
    unique_hashes = len(fold_map)
    print(f"Found {unique_hashes} unique shell_hashes in fold_map.")
    if unique_hashes < 1000:
        print("WARNING: Very few shell_hashes found. Check fold_map logic!")
    return fold_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0, help="Part index (0 to total_parts-1)")
    parser.add_argument("--total-parts", type=int, default=1, help="Total number of parts")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fold_map = build_fold_map()
    if not fold_map:
        print("Fold map is empty. Aborting.")
        return

    print("1. Parsing boards from Solvables...")
    records = parse_sok_files(BUCKET_DIR, fold_map)
    print(f"Loaded {len(records)} total boards.")
    
    # Sort deterministically and slice
    records.sort(key=lambda r: r['hash'])
    records = records[args.part::args.total_parts]
    print(f"Assigned {len(records)} boards to Part {args.part} of {args.total_parts}.")
    if len(records) == 0:
        print(f"Warning: No boards found in {BUCKET_DIR}. Make sure the dataset exists!")
        return
        
    print("2. Simulating paths with strict 60s timeout per board...")
    fold_datasets = {k: [] for k in range(1, 6)}
    
    # Intentar reanudar desde checkpoints existentes
    start_route_id = 0
    has_checkpoints = False
    for k in range(1, 6):
        out_path = os.path.join(OUTPUT_DIR, f"path_fold{k}_train_part{args.part}.pt")
        if os.path.exists(out_path):
            try:
                fold_datasets[k] = torch.load(out_path, map_location='cpu', weights_only=False)
                has_checkpoints = True
                if len(fold_datasets[k]) > 0:
                    start_route_id = max(start_route_id, fold_datasets[k][-1]['route_id'] + 1)
            except Exception as e:
                print(f"No se pudo cargar {out_path}: {e}")
                
    if has_checkpoints:
        print(f"Checkpoints detectados. Reanudando desde route_id = {start_route_id}")
        
    discard_count = 0
    route_id_counter = 0
    
    os.makedirs(os.path.dirname(OUTPUT_SOK), exist_ok=True)
    
    for record in tqdm(records):
        board_str = record['board_str']
        shell_hash = record['hash']
        
        # Si este tablero ya fue procesado en sesiones anteriores, saltarlo
        if route_id_counter < start_route_id:
            route_id_counter += 1
            continue
            
        if shell_hash not in fold_map:
            discard_count += 1
            continue
            
        fold_idx = fold_map[shell_hash]
        
        # Write single board to .sok
        with open(OUTPUT_SOK, "w") as f:
            f.write(f"{record['name']} - pushes:{record['pushes']}\n")
            f.write(f"{board_str}\n\n")
            
        # Run batch_solver for this single board
        cmd = ["../build/batch_solver", OUTPUT_SOK, "hungarian", OUTPUT_TSV]
        try:
            # 60 seconds strict timeout per board
            result = subprocess.run(cmd, check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.TimeoutExpired:
            continue
        except subprocess.CalledProcessError as e:
            print(f"Error executing batch_solver: {e}. Stderr: {e.stderr}")
            continue
        except Exception as e:
            print(f"Error executing batch_solver: {e}")
            continue
            
        # Parse single-row TSV
        try:
            df = pd.read_csv(OUTPUT_TSV, sep='\t')
            if len(df) == 0: continue
            row = df.iloc[0]
            if row['Status'] != 'SOLVED' or row['LURD_Path'] == 'NONE': continue
            lurd = str(row['LURD_Path'])
        except Exception:
            continue
            
        # Simulate path and pair states
        states = simulate_path(board_str, lurd)
        n = len(states)
        for i in range(n):
            for j in range(i+1, n):
                s1, p1 = states[i]
                s2, p2 = states[j]
                t1 = encode_board(s1)
                t2 = encode_board(s2)
                
                fold_datasets[fold_idx].append({
                    "tensor1": t1, "pushes1": p1,
                    "tensor2": t2, "pushes2": p2,
                    "route_id": route_id_counter,
                    "shell_hash": shell_hash
                })
        route_id_counter += 1
        
        # Guardar checkpoint cada 1000 tableros procesados
        if route_id_counter % 1000 == 0:
            print(f"\n[Checkpoint] Guardando progreso intermedio (Tableros resueltos: {route_id_counter})...")
            for k in range(1, 6):
                out_path = os.path.join(OUTPUT_DIR, f"path_fold{k}_train_part{args.part}.pt")
                if len(fold_datasets[k]) > 0:
                    torch.save(fold_datasets[k], out_path)
            print("Checkpoint guardado exitosamente.")
        
    print(f"Discarded {discard_count} boards due to missing shell_hash in fold_map.")
    
    for k in range(1, 6):
        out_path = os.path.join(OUTPUT_DIR, f"path_fold{k}_train_part{args.part}.pt")
        print(f"Fold {k}: saving {len(fold_datasets[k])} pairs to {out_path}")
        torch.save(fold_datasets[k], out_path)
        
    print("Done!")

if __name__ == "__main__":
    main()
