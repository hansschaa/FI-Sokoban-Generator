import sys
import torch
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
from optuna_path_consistency import evaluate_model_inter_branch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SokobanSEResNetRegressor(dropout_p=0.1).to(device)

print("Starting evaluation...")
acc = evaluate_model_inter_branch(model, device, n_pairs=500)
print(f"Accuracy: {acc}")
