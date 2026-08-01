import os
import sys
import time
import glob
import torch

sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
from evaluate_inter_branch import get_valid_children
from prepare_path_consistency import encode_board

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SokobanSEResNetRegressor(dropout_p=0.1).to(device)
model.eval()

# 1. Grab 500 real boards from training_data/Solvables
SOK_DIR = "training_data/Solvables"
files = glob.glob(os.path.join(SOK_DIR, "**/*.sok"), recursive=True)[:500]

boards = []
for f in files:
    with open(f, 'r') as file:
        boards.append(file.read())

if not boards:
    print("No boards found!")
    sys.exit(1)

# Ensure we have 500
while len(boards) < 500:
    boards.extend(boards[:500 - len(boards)])

print(f"Loaded {len(boards)} boards. Starting timing...")

# 2. Evaluate 500 states
start = time.time()
total_pairs = 0

with torch.no_grad():
    for board_str in boards:
        children = get_valid_children(board_str)
        if len(children) < 2:
            continue
            
        s_opt = children[0]
        sub_children = children[1:]
        
        t_opt = encode_board(s_opt)
        pred_opt = model(torch.tensor(t_opt).unsqueeze(0).to(device)).item()
        
        for s_sub in sub_children:
            t_sub = encode_board(s_sub)
            pred_sub = model(torch.tensor(t_sub).unsqueeze(0).to(device)).item()
            total_pairs += 1

t = time.time() - start
print(f"Time for {len(boards)} states (evaluated {total_pairs} sibling pairs): {t:.2f} seconds")
print(f"Time per pair: {(t/max(1, total_pairs)):.4f} seconds")

