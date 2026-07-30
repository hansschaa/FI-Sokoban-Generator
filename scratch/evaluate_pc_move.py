import os
import torch
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "surrogate_models/results"

test_data = torch.load(os.path.join(RESULTS_DIR, "regressor_fold1_test.pt"), weights_only=False)
model_path = os.path.join(RESULTS_DIR, "path_consistency", "final_regressor_fold1.pt")
state = torch.load(model_path, map_location=device, weights_only=False)
model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)

if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()

groups_pure_move = defaultdict(lambda: ([], []))
with torch.no_grad():
    for item in test_data:
        x = item['tensor'].unsqueeze(0).to(device)
        pred_val = model(x).item()
        
        # Necesitamos p_mean y p_std para desnormalizar correctamente
        # ya que la red predice z-scores
        stats = torch.load(os.path.join(RESULTS_DIR, "regressor_fold1_stats.pt"), map_location='cpu', weights_only=True)
        p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]
        pred_desnorm = pred_val * p_std + p_mean
        pred_pushes = np.expm1(pred_desnorm)
        
        real_pushes = item['pushes_raw']
        sh = item['shell_hash']
        num_boxes = item.get('num_boxes', 0)
        if num_boxes == 0:
            num_boxes = int(item['tensor'][2].sum().item())
            
        groups_pure_move[(sh, num_boxes)][0].append(real_pushes)
        groups_pure_move[(sh, num_boxes)][1].append(pred_pushes)

rhos_move = []
for key, (real, pred) in groups_pure_move.items():
    if len(real) >= 3:
        rho, _ = spearmanr(real, pred)
        if not np.isnan(rho):
            rhos_move.append(rho)

print(f"Spearman intra-shell move: {np.mean(rhos_move):.4f}")
