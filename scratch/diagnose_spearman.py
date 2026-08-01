import os
import torch
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "surrogate_models/results"

test_data = torch.load(os.path.join(RESULTS_DIR, "regressor_fold1_test.pt"), weights_only=False)

model_path = "scratch/trial_34.pt"
state = torch.load(model_path, map_location=device, weights_only=False)
model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()

groups_fixed_player = defaultdict(lambda: ([], []))
groups_fixed_boxes = defaultdict(lambda: ([], []))

with torch.no_grad():
    for item in test_data:
        t = item['tensor']
        x = t.unsqueeze(0).to(device)
        pred_val = model(x).item()
        pred_pushes = np.exp(pred_val) - 1.0
        
        real_pushes = item['pushes_raw']
        sh = item['shell_hash']
        
        # Hash for player position (Channel 4)
        player_idx = torch.nonzero(t[4]).reshape(-1).tolist()
        player_key = tuple(player_idx)
        
        # Hash for box positions (Channel 2)
        boxes_idx = torch.nonzero(t[2]).reshape(-1).tolist()
        boxes_key = tuple(boxes_idx)
        
        groups_fixed_player[(sh, player_key)][0].append(real_pushes)
        groups_fixed_player[(sh, player_key)][1].append(pred_pushes)
        
        groups_fixed_boxes[(sh, boxes_key)][0].append(real_pushes)
        groups_fixed_boxes[(sh, boxes_key)][1].append(pred_pushes)

rhos_boxes_vary = []
for key, (real, pred) in groups_fixed_player.items():
    # Only keep groups where real pushes actually vary (otherwise spearman is NaN)
    if len(real) >= 3 and len(set(real)) > 1:
        rho, _ = spearmanr(real, pred)
        if not np.isnan(rho):
            rhos_boxes_vary.append(rho)

rhos_player_vary = []
for key, (real, pred) in groups_fixed_boxes.items():
    if len(real) >= 3 and len(set(real)) > 1:
        rho, _ = spearmanr(real, pred)
        if not np.isnan(rho):
            rhos_player_vary.append(rho)

print(f"Grupo 1: Jugador fijo, CAJAS VARÍAN -> media={np.mean(rhos_boxes_vary):.3f}, n_grupos={len(rhos_boxes_vary)}")
print(f"Grupo 2: Cajas fijas, JUGADOR VARÍA -> media={np.mean(rhos_player_vary):.3f}, n_grupos={len(rhos_player_vary)}")

