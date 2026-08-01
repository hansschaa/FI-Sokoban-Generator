import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SokobanSEResNetRegressor(dropout_p=0.1).to(device)

epochs = 10
batch_size = 256
n_samples = 50000

# Dummy data
x_board = torch.randn(batch_size, 4, 10, 10, device=device)
x_sibling = torch.randn(batch_size, 4, 10, 10, device=device)
y_target = torch.randn(batch_size, device=device)
target_rank = torch.full_like(y_target, -1, device=device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
huber_loss = nn.HuberLoss(delta=1.0)
ranking_loss = nn.MarginRankingLoss(margin=0.1)

print("Starting simulated training...")
start = time.time()

model.train()
for epoch in range(epochs):
    epoch_start = time.time()
    n_batches = n_samples // batch_size
    for _ in range(n_batches):
        optimizer.zero_grad()
        pred_opt = model(x_board).squeeze(-1)
        pred_sub = model(x_sibling).squeeze(-1)
        
        loss_huber = huber_loss(pred_opt, y_target)
        loss_rank = ranking_loss(pred_opt, pred_sub, target_rank)
        
        total_loss = loss_huber + 0.1 * loss_rank
        total_loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} took {time.time() - epoch_start:.2f} seconds")

total_time = time.time() - start
print(f"Total training time for {epochs} epochs: {total_time:.2f} seconds")
