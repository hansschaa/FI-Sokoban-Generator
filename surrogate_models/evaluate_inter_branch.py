import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.resnet import SokobanSEResNetRegressor
# We reuse encode_board from prepare_path_consistency
from prepare_path_consistency import encode_board, parse_sok_files, simulate_path

def get_valid_children(board_str):
    lines = [list(l) for l in board_str.splitlines()]
    H = len(lines)
    if H == 0: return []
    W = max(len(l) for l in lines)
    for i in range(H):
        lines[i] += [' '] * (W - len(lines[i]))
        
    px, py = -1, -1
    for r in range(H):
        for c in range(W):
            if lines[r][c] in ['@', '+']:
                px, py = r, c
                break
        if px != -1: break
        
    reachable = set()
    q = [(px, py)]
    visited = set([(px, py)])
    while q:
        x, y = q.pop(0)
        reachable.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                if lines[nx][ny] in [' ', '.', '@', '+']:
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    
    children = []
    for r in range(H):
        for c in range(W):
            if lines[r][c] in ['$', '*']:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    px_req, py_req = r - dx, c - dy
                    nx, ny = r + dx, c + dy
                    if (px_req, py_req) in reachable:
                        if 0 <= nx < H and 0 <= ny < W and lines[nx][ny] in [' ', '.']:
                            new_lines = [row[:] for row in lines]
                            new_lines[px][py] = ' ' if new_lines[px][py] == '@' else '.'
                            new_lines[r][c] = '@' if lines[r][c] == '$' else '+'
                            new_lines[nx][ny] = '$' if lines[nx][ny] == ' ' else '*'
                            child_str = "\n".join("".join(row).rstrip() for row in new_lines)
                            children.append(child_str)
    return children

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = SokobanSEResNetRegressor().to(device)
    model_path = "results/path_consistency/consistent_regressor.pt"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Fallback to fold1.")
        model_path = "results/final_regressor_fold1.pt"
        
    print(f"Evaluating {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load original TSV and .sok for paths
    TSV_FILE = "../scratch/path_consistency_results.tsv"
    SOK_DIR = "../sokoban_dataset_buckets"
    
    if not os.path.exists(TSV_FILE):
        print("TSV file not found!")
        return
        
    records = parse_sok_files(SOK_DIR, max_total=500) # subset for evaluation
    board_map = {r['name']: r['board_str'] for r in records}
    
    df = pd.read_csv(TSV_FILE, sep='\t')
    
    total_pairs = 0
    correct_pairs = 0
    
    # Batch variables
    batch_tensors = []
    batch_info = [] # (pair_idx, is_optimal)
    
    print("Evaluating inter-branch predictions...")
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=min(len(df), 500)):
            if row['Status'] != 'SOLVED': continue
            if row['LURD_Path'] == 'NONE': continue
            
            name = str(row['LevelName']).split(' - ')[0].strip()
            if name not in board_map: continue
            
            board_str = board_map[name]
            lurd = row['LURD_Path']
            
            # states contains (board_str, remaining_pushes)
            # states[0] is initial, states[-1] is goal
            states = simulate_path(board_str, lurd)
            
            for i in range(len(states) - 1):
                s_curr, _ = states[i]
                s_opt, _ = states[i+1] # The optimal child
                
                children = get_valid_children(s_curr)
                # Keep only suboptimal children
                sub_children = [c for c in children if c != s_opt]
                
                if not sub_children: continue
                
                # Encode optimal
                t_opt = encode_board(s_opt)
                pred_opt = model(torch.tensor(t_opt).unsqueeze(0).to(device)).item()
                
                for s_sub in sub_children:
                    t_sub = encode_board(s_sub)
                    pred_sub = model(torch.tensor(t_sub).unsqueeze(0).to(device)).item()
                    
                    total_pairs += 1
                    if pred_opt < pred_sub:
                        correct_pairs += 1
                        
    if total_pairs > 0:
        acc = correct_pairs / total_pairs * 100
        print(f"Inter-branch accuracy: {acc:.2f}% ({correct_pairs}/{total_pairs})")
    else:
        print("No pairs found.")

if __name__ == "__main__":
    main()
