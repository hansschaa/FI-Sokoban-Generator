import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
from train_final_path_consistency import PathConsistencyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters from Trial 34
lr = 0.0000951
alpha = 0.395
margin = 0.179
weight_decay = 1e-4
dropout_p = 0.1
batch_size = 256
epochs = 10

print("Initializing model...")
model = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
state_dict = torch.load('surrogate_models/results/final_regressor_fold1.pt', map_location=device, weights_only=False)
model.load_state_dict(state_dict)

dataset = PathConsistencyDataset(1, augment=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
huber_loss = nn.HuberLoss(delta=1.0)
ranking_loss = nn.MarginRankingLoss(margin=margin)

print("Starting training...")
for epoch in range(epochs):
    t0 = time.time()
    model.train()
    for batch in dataloader:
        x_board = batch['tensor1'].to(device)
        p1_raw = batch['pushes1'].float().to(device)
        x_sibling = batch['tensor2'].to(device)
        
        y_target = (torch.log1p(p1_raw) - 2.45) / 1.05
        
        optimizer.zero_grad()
        
        pred_opt = model(x_board).squeeze(-1)
        pred_sub = model(x_sibling).squeeze(-1)
        
        loss_huber = huber_loss(pred_opt, y_target)
        target_rank = torch.full_like(pred_opt, -1)
        loss_rank = ranking_loss(pred_opt, pred_sub, target_rank)
        
        total_loss = loss_huber + alpha * loss_rank
        total_loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} done in {time.time()-t0:.1f}s")

torch.save(model.state_dict(), 'scratch/trial_34.pt')
print("Model saved to scratch/trial_34.pt")

from optuna_path_consistency import evaluate_model_inter_branch
print("Evaluating inter-branch metric...")
acc = evaluate_model_inter_branch(model, device, n_pairs=500)
print(f"Acc: {acc}")
