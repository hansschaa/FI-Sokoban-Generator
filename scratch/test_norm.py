import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load a checkpoint
ckpt = torch.load("results/path_consistency/ckpt_regressor_fold1.pt", map_location='cpu')

from models.resnet import SokobanSEResNetRegressor
model = SokobanSEResNetRegressor(dropout_p=0.02)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

stats = torch.load("results/regressor_fold1_stats.pt", map_location='cpu')
p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]
print(f"p_mean: {p_mean}, p_std: {p_std}")

val_data = torch.load("results/regressor_fold1_val.pt", map_location='cpu')
print(f"Loaded {len(val_data)} val samples")

print("\n--- TEST PREDICCIONES ---")
with torch.no_grad():
    for i in range(5):
        t = val_data[i]['tensor'].unsqueeze(0).float()
        p_raw = val_data[i]['pushes_raw']
        p_norm_expected = val_data[i]['pushes_norm']
        p_pred = model(t).squeeze()
        
        p_desnorm = p_pred * p_std + p_mean
        p_desnorm_real = torch.expm1(p_desnorm)
        
        print(f"Sample {i}: Real Pushes={p_raw:.1f} (norm {p_norm_expected:.4f}) | Pred norm={p_pred:.4f} | Pred desnorm pushes={p_desnorm_real:.1f}")
