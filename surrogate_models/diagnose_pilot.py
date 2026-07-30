import torch
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from models.resnet import SokobanSEResNetRegressor
import os

def diagnose_spearman(model, device):
    test_data = torch.load("results/regressor_fold1_test.pt", weights_only=False, map_location='cpu')
    groups_fixed_player = defaultdict(lambda: ([], []))
    groups_fixed_boxes = defaultdict(lambda: ([], []))
    
    with torch.no_grad():
        for item in test_data:
            t = item['tensor']
            x = t.unsqueeze(0).to(device)
            pred_pushes = np.exp(model(x).item()) - 1.0
            
            real_pushes = item['pushes_raw']
            sh = item['shell_hash']
            player_key = tuple(torch.nonzero(t[4]).reshape(-1).tolist())
            boxes_key = tuple(torch.nonzero(t[2]).reshape(-1).tolist())
            
            groups_fixed_player[(sh, player_key)][0].append(real_pushes)
            groups_fixed_player[(sh, player_key)][1].append(pred_pushes)
            groups_fixed_boxes[(sh, boxes_key)][0].append(real_pushes)
            groups_fixed_boxes[(sh, boxes_key)][1].append(pred_pushes)

    rhos_boxes_vary = []
    for key, (real, pred) in groups_fixed_player.items():
        if len(real) >= 3 and len(set(real)) > 1:
            rho, _ = spearmanr(real, pred)
            if not np.isnan(rho): rhos_boxes_vary.append(rho)

    rhos_player_vary = []
    for key, (real, pred) in groups_fixed_boxes.items():
        if len(real) >= 3 and len(set(real)) > 1:
            rho, _ = spearmanr(real, pred)
            if not np.isnan(rho): rhos_player_vary.append(rho)
            
    return np.mean(rhos_boxes_vary), np.mean(rhos_player_vary)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SokobanSEResNetRegressor(dropout_p=0.02).to(device)
    model.load_state_dict(torch.load('results/path_consistency/final_regressor_fold1.pt', map_location=device, weights_only=False))

    print("Calculando Spearman Intra-shell en el set de Test...")
    rho_boxes, rho_player = diagnose_spearman(model, device)
    print(f"  Spearman (Fixed Player, Boxes Vary): {rho_boxes:.3f}")
    print(f"  Spearman (Fixed Boxes, Player Vary): {rho_player:.3f}")
