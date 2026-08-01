import torch
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SokobanSEResNetRegressor().to(device)
model.load_state_dict(torch.load('surrogate_models/results/final_regressor_fold1.pt', map_location=device, weights_only=False))

model.train()

# Dummy batch: batch_size=256, channels=6, H=10, W=10
x_board = torch.randn(256, 6, 10, 10).to(device)
x_sibling = torch.randn(256, 6, 10, 10).to(device)

pred_opt = model(x_board).squeeze(-1)
pred_sub = model(x_sibling).squeeze(-1)

# Just compute the Ranking Loss
target_rank = torch.full_like(pred_opt, -1)
loss_rank = torch.nn.MarginRankingLoss(margin=0.179)(pred_opt, pred_sub, target_rank)

loss_rank.backward()

# Find first conv layer
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        conv_name = name
        break

grad = model.state_dict(keep_vars=True)[f"{conv_name}.weight"].grad
print("Ranking Loss Gradient L1 Norm per channel:")
for c in range(grad.shape[1]):
    print(f"  Channel {c}: {grad[:, c, :, :].abs().mean().item():.8f}")
