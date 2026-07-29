import os
import glob
import re
import hashlib
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import subprocess

K_STEPS = 4
MAX_SAMPLES = 1000
BUCKET_DIR = "../sokoban_dataset_buckets"
OUTPUT_DIR = "results/path_consistency"
OUTPUT_SOK = "../scratch/path_consistency_sample.sok"
OUTPUT_TSV = "../scratch/path_consistency_results.tsv"

def get_bucket(pushes):
    if pushes <= 10: return "1_to_10"
    if pushes > 100: return "101_plus"
    lower = ((pushes - 1) // 10) * 10 + 1
    upper = lower + 9
    return f"{lower}_to_{upper}"

def parse_sok_files(directory, max_total=MAX_SAMPLES):
    records = []
    files = [
        "21_to_30.sok",
        "31_to_40.sok",
        "41_to_50.sok",
        "51_to_60.sok"
    ]
    
    for fname in files:
        fpath = os.path.join(directory, fname)
        if not os.path.exists(fpath): continue
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
            level_name = m.group(1)
            stats_str = m.group(2)
            
            pushes = -1
            if "pushes:" in stats_str:
                for token in stats_str.split():
                    if token.startswith("pushes:"):
                        pushes = int(token.split(":")[1])
            elif stats_str.isdigit():
                pushes = int(stats_str)
            
            if pushes <= 0: continue
            
            board_str = "\n".join(board_lines)
            board_hash = hashlib.sha256(board_str.encode()).hexdigest()
            records.append({
                "hash": board_hash,
                "name": level_name,
                "board_str": board_str,
                "pushes": pushes,
                "bucket": get_bucket(pushes)
            })
            if len(records) >= max_total:
                return records
    return records

def simulate_path(board_str, lurd_path):
    lines = [list(l) for l in board_str.splitlines()]
    px, py = -1, -1
    for r, row in enumerate(lines):
        for c, char in enumerate(row):
            if char in ['@', '+']:
                px, py = r, c
                break
        if px != -1: break
        
    states = []
    states.append(("\n".join("".join(row) for row in lines), len(lurd_path.replace('u','').replace('d','').replace('l','').replace('r',''))))
    
    dirs = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1),
            'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    
    remaining_pushes = states[0][1]
    
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
            remaining_pushes -= 1
            
        # Move player
        p_char = lines[px][py]
        lines[px][py] = ' ' if p_char == '@' else '.'
        n_char = lines[nx][ny]
        lines[nx][ny] = '@' if n_char in [' ', '-'] else '+'
        
        px, py = nx, ny
        
        if is_push:
            states.append(("\n".join("".join(row) for row in lines), remaining_pushes))
            
    return states

# --- TENSOR ENCODING ---
def flood_fill_exterior(char_matrix):
    H, W = char_matrix.shape
    visited = np.zeros((H, W), dtype=bool)
    q = []
    for r in range(H):
        q.append((r, 0)); q.append((r, W-1))
    for c in range(W):
        q.append((0, c)); q.append((H-1, c))
    
    exterior = np.zeros((H, W), dtype=bool)
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        if r < 0 or r >= H or c < 0 or c >= W: continue
        if visited[r, c]: continue
        visited[r, c] = True
        
        if char_matrix[r, c] == '#': continue
            
        exterior[r, c] = True
        q.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
    return exterior

def encode_board(board_str):
    lines = board_str.splitlines()
    H = len(lines)
    W = max(len(l) for l in lines)
    
    char_matrix = np.full((H, W), ' ', dtype=str)
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            char_matrix[r, c] = char
            
    exterior = flood_fill_exterior(char_matrix)
    tensor = np.zeros((6, H, W), dtype=np.float32)
    
    for r in range(H):
        for c in range(W):
            ch = char_matrix[r, c]
            if ch == '#':
                tensor[0, r, c] = 1.0 # Canal 0: Pared
            if ch in ['.', '*', '+']:
                tensor[1, r, c] = 1.0 # Canal 1: Metas
            if ch in ['$', '*']:
                tensor[2, r, c] = 1.0 # Canal 2: Cajas
            if ch in ['@', '+']:
                tensor[3, r, c] = 1.0 # Canal 3: Jugador
            # Canal 4 is deadlock_mask (zeros for now)
            if not exterior[r, c] and ch != '#':
                tensor[5, r, c] = 1.0 # Canal 5: Interior caminable
                
    # Pad to 25x25 (since model expects fixed or max shape? Wait, the model uses AdaptiveAvgPool2d(1), so any size works!
    # But wait, original code might pad to 25x25. 
    # Let's pad to 25x25 just to be safe and consistent with original data.
    padded_tensor = np.zeros((6, 25, 25), dtype=np.float32)
    h_min = min(H, 25)
    w_min = min(W, 25)
    padded_tensor[:, :h_min, :w_min] = tensor[:, :h_min, :w_min]
    return padded_tensor

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Parsing boards...")
    records = parse_sok_files(BUCKET_DIR)
    print(f"Loaded {len(records)} boards.")
    if len(records) == 0:
        print(f"Warning: No boards found in {BUCKET_DIR}. Make sure the dataset exists!")
        return
        
    print("2. Writing to temporary .sok file...")
    os.makedirs(os.path.dirname(OUTPUT_SOK), exist_ok=True)
    with open(OUTPUT_SOK, "w") as f:
        for r in records:
            f.write(f"{r['name']} - pushes:{r['pushes']}\n")
            f.write(f"{r['board_str']}\n\n")
            
    print("3. Skipping batch_solver (already ran)...")
    # cmd = ["../build/batch_solver", OUTPUT_SOK, "hungarian", OUTPUT_TSV]
    # subprocess.run(cmd, check=True)
    
    print("4. Parsing TSV and simulating paths...")
    df = pd.read_csv(OUTPUT_TSV, sep='\t')
    
    # Create lookup map
    board_map = {r['name']: r['board_str'] for r in records}
    
    dataset = []
    route_id_counter = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if row['Status'] != 'SOLVED': continue
        if row['LURD_Path'] == 'NONE': continue
        
        name = str(row['LevelName']).split(' - ')[0].strip()
        if name not in board_map: continue
        
        board_str = board_map[name]
        lurd = row['LURD_Path']
        
        # Simulate pushes only! Our states array contains ONLY the board after a PUSH.
        # So we can sample every K_STEPS pushes.
        states = simulate_path(board_str, lurd)
        
        # states contains (board_str, remaining_pushes)
        # Sample every K_STEPS
        sampled_states = states[::K_STEPS]
        if states[-1] not in sampled_states:
            sampled_states.append(states[-1]) # always include the goal state (0 pushes)
            
        if len(sampled_states) < 2: continue
        
        route_id = f"route_{route_id_counter}"
        route_id_counter += 1
        
        for b_str, rem_pushes in sampled_states:
            tensor = encode_board(b_str)
            dataset.append({
                "tensor": torch.tensor(tensor),
                "pushes": rem_pushes,
                "route_id": route_id
            })
            
    print(f"Generated {len(dataset)} states across {route_id_counter} routes.")
    
    print("5. Saving PyTorch dataset...")
    torch.save(dataset, os.path.join(OUTPUT_DIR, "path_consistency_train.pt"))
    print("Done!")

if __name__ == "__main__":
    main()
