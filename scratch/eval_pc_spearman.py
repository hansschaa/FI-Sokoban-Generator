import os
import torch
import numpy as np
from scipy.stats import spearmanr
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

RESULTS_DIR = "surrogate_models/results"
model_path = os.path.join(RESULTS_DIR, "path_consistency", "consistent_regressor.pt")
test_data_path = os.path.join(RESULTS_DIR, "regressor_fold1_test.pt")

print(f"Loading model: {model_path}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SokobanSEResNetRegressor().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Loading test data...")
test_data = torch.load(test_data_path, map_location=device, weights_only=False)

groups = {}
for item in test_data:
    if 'shell_hash' in item:
        t = item['tensor']
        num_boxes = int(t[2].sum().item())
        key = (item['shell_hash'], num_boxes)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

valid_move_groups = [g for g in groups.values() if len(g) > 1]
print(f"Found {len(valid_move_groups)} valid groups for MOVE.")

move_corrs = []
for g in valid_move_groups:
    real = [item['pushes_raw'] for item in g]
    if len(set(real)) > 1:
        tensors = torch.stack([item['tensor'] for item in g]).to(device)
        with torch.no_grad():
            pred = model(tensors).squeeze(-1).cpu().numpy()
        corr, _ = spearmanr(real, pred)
        if not np.isnan(corr):
            move_corrs.append(corr)

if move_corrs:
    print(f"Intra-shell MOVE Spearman: {np.mean(move_corrs):.4f} (over {len(move_corrs)} valid groups)")

box_groups = {}
for item in test_data:
    if 'shell_hash' in item:
        t = item['tensor']
        player_pos = tuple((t[4] == 1).nonzero(as_tuple=False).flatten().tolist())
        key = (item['shell_hash'], player_pos)
        if key not in box_groups:
            box_groups[key] = []
        box_groups[key].append(item)

valid_box_groups = [g for g in box_groups.values() if len(g) > 1]
print(f"Found {len(valid_box_groups)} valid groups for ADD/REMOVE.")

box_corrs = []
for g in valid_box_groups:
    real = [item['pushes_raw'] for item in g]
    if len(set(real)) > 1:
        tensors = torch.stack([item['tensor'] for item in g]).to(device)
        with torch.no_grad():
            pred = model(tensors).squeeze(-1).cpu().numpy()
        corr, _ = spearmanr(real, pred)
        if not np.isnan(corr):
            box_corrs.append(corr)
            
if box_corrs:
    print(f"Intra-shell ADD/REMOVE Spearman: {np.mean(box_corrs):.4f} (over {len(box_corrs)} valid groups)")
