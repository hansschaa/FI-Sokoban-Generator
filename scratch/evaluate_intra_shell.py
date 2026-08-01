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

# Agrupar por shell
groups = defaultdict(lambda: ([], []))

print("Procesando y prediciendo...")
with torch.no_grad():
    for item in test_data:
        x = item['tensor'].unsqueeze(0).to(device)
        pred_val = model(x).item()
        
        # El modelo predice target. Asumiendo que el modelo actual es el baseline de MAE logaritmico normal:
        # pred = exp(out) - 1
        # Si era el modelo residual: pred = exp(out) - 1 + hungarian_lb
        # Voy a usar el código que entrenó final_regressor_fold1.pt. Fue sin residual.
        
        pred_pushes = np.exp(pred_val) - 1.0
        
        real_pushes = item['pushes_raw']
        sh = item['shell_hash']
        
        groups[sh][0].append(real_pushes)
        groups[sh][1].append(pred_pushes)

rhos = []
rhos_by_diff = defaultdict(list)

for shell_hash, (real, pred) in groups.items():
    if len(real) >= 3:
        rho, _ = spearmanr(real, pred)
        if not np.isnan(rho):
            rhos.append(rho)
            avg_diff = np.mean(real)
            bucket = int(avg_diff // 20) * 20
            rhos_by_diff[bucket].append(rho)

print(f"\n=== Spearman INTRA-SHELL ===\nMedia global: {np.mean(rhos):.3f} (sobre {len(rhos)} shells con >=3 variantes)\n")
print("Desglose por dificultad promedio del shell:")
for b in sorted(rhos_by_diff.keys()):
    vals = rhos_by_diff[b]
    print(f"  Rango {b:3d} a {b+19:3d}: Sp = {np.mean(vals):.3f} (n={len(vals)} shells)")
