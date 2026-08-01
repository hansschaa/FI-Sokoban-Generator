import torch
import sys
sys.path.append('surrogate_models')
sys.path.append('scratch')
from models.resnet import SokobanSEResNetRegressor
from run_pilot import diagnose_spearman

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SokobanSEResNetRegressor(dropout_p=0.02).to(device)
model.load_state_dict(torch.load('surrogate_models/results/path_consistency/final_regressor_fold1.pt', map_location=device, weights_only=False))

rho_boxes, rho_player = diagnose_spearman(model, device)
print(f"  Spearman (Fixed Player, Boxes Vary): {rho_boxes:.3f}")
print(f"  Spearman (Fixed Boxes, Player Vary): {rho_player:.3f}")
