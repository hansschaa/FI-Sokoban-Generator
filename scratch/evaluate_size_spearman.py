import os
import torch
import numpy as np
from scipy.stats import spearmanr
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
from train_final_surrogates import RegressorDataset
from torch.utils.data import DataLoader

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

# Haremos las predicciones
predictions = []
targets = []
buckets = []
sizes = []

print("Procesando y prediciendo...")
dataset = RegressorDataset(test_data)
loader = DataLoader(dataset, batch_size=256, shuffle=False)

with torch.no_grad():
    for x, y, _, _ in loader:
        x = x.to(device)
        pred = torch.exp(model(x)) - 1.0
        predictions.extend(pred.cpu().numpy().flatten())
        targets.extend(y.numpy())
        
# Calcular tamaños
for item in test_data:
    buckets.append(item['bucket'])
    tensor = item['tensor']
    # Canal 1 es piso alcanzable
    rows, cols = np.where(tensor[1].numpy() == 1)
    if len(rows) > 0:
        real_h = max(rows) - min(rows) + 3 # +2 para los muros, +1 porque min/max son inclusivos
        real_w = max(cols) - min(cols) + 3
        max_dim = max(real_h, real_w)
    else:
        max_dim = 25
    sizes.append(max_dim)

predictions = np.array(predictions)
targets = np.array(targets)
sizes = np.array(sizes)
buckets = np.array(buckets)

# Split y evaluar
print("\n=== Spearman por tamaño en buckets medios ===")
for target_bucket in ["31_to_40", "41_to_50", "51_to_60", "61_to_70"]:
    mask_bucket = (buckets == target_bucket)
    if not np.any(mask_bucket): continue
    
    t_bucket = targets[mask_bucket]
    p_bucket = predictions[mask_bucket]
    s_bucket = sizes[mask_bucket]
    
    mask_small = s_bucket <= 12
    mask_large = s_bucket > 12
    
    sp_all = spearmanr(t_bucket, p_bucket).correlation if len(t_bucket) > 1 else float('nan')
    
    if np.sum(mask_small) > 1:
        sp_small = spearmanr(t_bucket[mask_small], p_bucket[mask_small]).correlation
    else:
        sp_small = float('nan')
        
    if np.sum(mask_large) > 1:
        sp_large = spearmanr(t_bucket[mask_large], p_bucket[mask_large]).correlation
    else:
        sp_large = float('nan')
        
    print(f"\nBucket {target_bucket} (Total n={len(t_bucket)}): Global Sp = {sp_all:.3f}")
    print(f"  -> Pequeños (<=12): n={np.sum(mask_small):4d} | Sp = {sp_small:.3f}")
    print(f"  -> Grandes   (>12): n={np.sum(mask_large):4d} | Sp = {sp_large:.3f}")

