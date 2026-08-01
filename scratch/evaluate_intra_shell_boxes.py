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

print("Cargando test set...")
test_data = torch.load(os.path.join(RESULTS_DIR, "regressor_fold1_test.pt"), weights_only=False)

print("Cargando modelo SE-ResNet...")
model_path = os.path.join(RESULTS_DIR, "final_regressor_fold1.pt")
state = torch.load(model_path, map_location=device, weights_only=False)
model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()

shells = defaultdict(list)

print("Procesando y prediciendo...")
with torch.no_grad():
    for item in test_data:
        x = item['tensor'].unsqueeze(0).to(device)
        pred_val = model(x).item()
        pred_pushes = np.exp(pred_val) - 1.0
        
        real_pushes = item['pushes_raw']
        sh = item['shell_hash']
        num_boxes = item.get('num_boxes', 0)
        
        # Si num_boxes no existe, contamos las cajas en el tensor (canal 2)
        if num_boxes == 0:
            num_boxes = int(item['tensor'][2].sum().item())
            
        shells[sh].append((num_boxes, real_pushes, pred_pushes))

same_rhos = []
diff_rhos = []

for shell_hash, variants in shells.items():
    if len(variants) < 3:
        continue
        
    box_counts = set(v[0] for v in variants)
    real = [v[1] for v in variants]
    p = [v[2] for v in variants]
    
    rho, _ = spearmanr(real, p)
    if np.isnan(rho):
        continue
        
    if len(box_counts) == 1:
        same_rhos.append(rho)
    else:
        diff_rhos.append(rho)

print(f"\nIntra-shell, MISMA cantidad de cajas (tipo 'move'): media={np.mean(same_rhos):.3f}, n_shells={len(same_rhos)}")
if len(diff_rhos) > 0:
    print(f"Intra-shell, DISTINTA cantidad de cajas (tipo 'add/remove'): media={np.mean(diff_rhos):.3f}, n_shells={len(diff_rhos)}")
else:
    print("No hay shells con distinta cantidad de cajas en el test set (n=0).")

