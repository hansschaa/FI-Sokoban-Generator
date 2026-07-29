import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import copy
from models.resnet import SokobanSEResNetRegressor

class PathConsistencyDataset(Dataset):
    def __init__(self, pt_file):
        self.pairs = torch.load(pt_file, weights_only=False)
                    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        item = self.pairs[idx]
        return {
            "tensor1": item["tensor1"],
            "pushes1": item["pushes1"],
            "tensor2": item["tensor2"],
            "pushes2": item["pushes2"]
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SokobanSEResNetRegressor().to(device)
    
    # Check what models exist, load fold1 or production
    model_path = "results/final_regressor_fold1.pt"
    stats_path = "results/regressor_fold1_stats.pt"
    if not os.path.exists(model_path):
        model_path = "results/production_regressor.pt"
        stats_path = "results/production_regressor_stats.pt"
        
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Starting from scratch!")
        
    stats = torch.load(stats_path, map_location='cpu')
    p_mean = stats['pushes_mean']
    p_std = stats['pushes_std']
    print(f"Loaded stats: pushes_mean={p_mean:.4f}, pushes_std={p_std:.4f}")
    
    dataset_path = "results/path_consistency/path_fold1_train.pt"
    print(f"Loading dataset from {dataset_path}...")
    dataset = PathConsistencyDataset(dataset_path)
    print(f"Generated {len(dataset)} consistency pairs.")
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    mse_loss = nn.MSELoss()
    margin_loss = nn.MarginRankingLoss(margin=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1):
        total_loss = 0
        total_mse = 0
        total_margin = 0
        
        for batch in loader:
            optimizer.zero_grad()
            t1, p1 = batch["tensor1"].to(device), batch["pushes1"].float().to(device)
            t2, p2 = batch["tensor2"].to(device), batch["pushes2"].float().to(device)
            
            # Normalize the pushes
            p1_norm = (torch.log1p(p1) - p_mean) / p_std
            p2_norm = (torch.log1p(p2) - p_mean) / p_std
            
            pred1 = model(t1).squeeze()
            pred2 = model(t2).squeeze()
            
            loss_mse1 = mse_loss(pred1, p1_norm)
            loss_mse2 = mse_loss(pred2, p2_norm)
            
            # Since state 1 is ALWAYS earlier in the route, it has MORE pushes than state 2
            # So we want pred1 > pred2.
            y = torch.ones_like(pred1)
            l_margin = margin_loss(pred1, pred2, y)
            
            # Combine losses
            loss = loss_mse1 + loss_mse2 + 2.0 * l_margin
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += (loss_mse1.item() + loss_mse2.item())
            total_margin += l_margin.item()
            
        avg_loss = total_loss/len(loader)
        avg_mse = total_mse/len(loader)
        avg_margin = total_margin/len(loader)
        
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Margin: {avg_margin:.4f})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            
    out_path = "results/path_consistency/consistent_regressor.pt"
    torch.save(best_state, out_path)
    print(f"Saved fine-tuned model to {out_path}")

if __name__ == "__main__":
    train()
