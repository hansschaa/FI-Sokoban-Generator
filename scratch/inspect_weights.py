import torch
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

model_orig = SokobanSEResNetRegressor()
model_orig.load_state_dict(torch.load('surrogate_models/results/final_regressor_fold1.pt', map_location='cpu', weights_only=False))

model_34 = SokobanSEResNetRegressor()
model_34.load_state_dict(torch.load('scratch/trial_34.pt', map_location='cpu'))

# Find first conv layer
# Usually it's model.conv1 or something similar. Let's just iterate over named_modules to find the first Conv2d
for name, module in model_orig.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        conv_name = name
        break

print(f"First conv layer: {conv_name}")
w_orig = model_orig.state_dict()[f"{conv_name}.weight"]
w_34 = model_34.state_dict()[f"{conv_name}.weight"]

print("Original Weights L1 Norm per channel:")
for c in range(w_orig.shape[1]):
    print(f"  Channel {c}: {w_orig[:, c, :, :].abs().mean().item():.6f}")

print("\nTrial 34 Weights L1 Norm per channel:")
for c in range(w_34.shape[1]):
    print(f"  Channel {c}: {w_34[:, c, :, :].abs().mean().item():.6f}")

diff = (w_34 - w_orig).abs().mean(dim=(0,2,3))
print("\nAbsolute Difference (Change in Weights) per channel:")
for c in range(len(diff)):
    print(f"  Channel {c}: {diff[c].item():.6f}")

