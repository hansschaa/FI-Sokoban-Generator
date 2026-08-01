import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import argparse
import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import pandas as pd

sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
from train_final_path_consistency import PathConsistencyDataset
from evaluate_inter_branch import get_valid_children
from prepare_path_consistency import encode_board, simulate_path

def evaluate_model_inter_branch(model, device, n_pairs=500):
    model.eval()
    TSV_FILE = "scratch/path_consistency_results.tsv"
    if not os.path.exists(TSV_FILE): return 0.0
    
    df = pd.read_csv(TSV_FILE, sep='\t')
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    board_map = {}
    for src in ["scratch/path_consistency_sample.sok", "sok_files/benchmark_stratified_heldout.sok"]:
        if os.path.exists(src):
            with open(src, 'r') as f: lines = f.readlines()
            current_name = None
            current_board = []
            for line in lines:
                line = line.rstrip()
                if ' - pushes:' in line or ' - moves:' in line:
                    if current_name and current_board: board_map[current_name] = '\n'.join(current_board)
                    current_name = line.split(' - ')[0].strip()
                    current_board = []
                elif line: current_board.append(line)
            if current_name and current_board: board_map[current_name] = '\n'.join(current_board)
            
    total_pairs, correct_pairs = 0, 0
    with torch.no_grad():
        for idx, row in df.iterrows():
            if total_pairs >= n_pairs: break
            if row['Status'] != 'SOLVED' or row['LURD_Path'] == 'NONE': continue
            name = str(row['LevelName']).split(' - ')[0].strip()
            if name not in board_map: continue
            
            states = simulate_path(board_map[name], row['LURD_Path'])
            for i in range(len(states) - 1):
                if total_pairs >= n_pairs: break
                s_curr = states[i][0]
                s_opt = states[i+1][0]
                children = get_valid_children(s_curr)
                sub_children = [c for c in children if c != s_opt]
                if not sub_children: continue
                
                pred_opt = model(torch.tensor(encode_board(s_opt)).unsqueeze(0).to(device)).item()
                for s_sub in sub_children:
                    pred_sub = model(torch.tensor(encode_board(s_sub)).unsqueeze(0).to(device)).item()
                    total_pairs += 1
                    if pred_opt < pred_sub: correct_pairs += 1
                    if total_pairs >= n_pairs: break
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def diagnose_spearman(model, device):
    test_data = torch.load("surrogate_models/results/regressor_fold1_test.pt", weights_only=False)
    groups_fixed_player = defaultdict(lambda: ([], []))
    groups_fixed_boxes = defaultdict(lambda: ([], []))
    
    with torch.no_grad():
        for item in test_data:
            t = item['tensor']
            x = t.unsqueeze(0).to(device)
            pred_pushes = np.exp(model(x).item()) - 1.0
            
            real_pushes = item['pushes_raw']
            sh = item['shell_hash']
            player_key = tuple(torch.nonzero(t[4]).reshape(-1).tolist())
            boxes_key = tuple(torch.nonzero(t[2]).reshape(-1).tolist())
            
            groups_fixed_player[(sh, player_key)][0].append(real_pushes)
            groups_fixed_player[(sh, player_key)][1].append(pred_pushes)
            groups_fixed_boxes[(sh, boxes_key)][0].append(real_pushes)
            groups_fixed_boxes[(sh, boxes_key)][1].append(pred_pushes)

    rhos_boxes_vary = []
    for key, (real, pred) in groups_fixed_player.items():
        if len(real) >= 3 and len(set(real)) > 1:
            rho, _ = spearmanr(real, pred)
            if not np.isnan(rho): rhos_boxes_vary.append(rho)

    rhos_player_vary = []
    for key, (real, pred) in groups_fixed_boxes.items():
        if len(real) >= 3 and len(set(real)) > 1:
            rho, _ = spearmanr(real, pred)
            if not np.isnan(rho): rhos_player_vary.append(rho)
            
    return np.mean(rhos_boxes_vary), np.mean(rhos_player_vary)

def run_pilot(alpha, margin, epochs):
    with open("surrogate_models/results/best_hparams.json", "r") as f:
        cfg = json.load(f)["params"]

    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p = cfg["dropout_p"]
    batch_size = int(cfg["batch_size"]) // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load('surrogate_models/results/final_regressor_fold1.pt', map_location=device, weights_only=False))
    
    dataset = PathConsistencyDataset(1, augment=True, max_route_distance=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    margin_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
    huber_loss = nn.HuberLoss(reduction='none')
    
    print(f"\n--- Running Pilot for alpha={alpha}, margin={margin}, epochs={epochs}, lr={lr} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in dataloader:
            t1 = batch['tensor1'].to(device)
            p1 = batch['pushes1'].float().to(device)
            t2 = batch['tensor2'].to(device)
            p2 = batch['pushes2'].float().to(device)
            weights = batch['weight'].to(device)
            
            p_mean, p_std = 2.45, 1.05
            p1_norm = (torch.log1p(p1) - p_mean) / p_std
            p2_norm = (torch.log1p(p2) - p_mean) / p_std
            
            optimizer.zero_grad()
            combined = torch.cat([t1, t2], dim=0)
            pred_combined = model(combined).squeeze(-1)
            pred1, pred2 = pred_combined.split(t1.size(0))
            
            loss_huber1 = huber_loss(pred1, p1_norm)
            loss_huber2 = huber_loss(pred2, p2_norm)
            loss_huber_batch = ((loss_huber1 + loss_huber2) / 2.0 * weights).mean()
            
            y = torch.ones_like(pred1)
            loss_margin_batch = (margin_loss(pred1, pred2, y) * weights).mean()
            
            loss = (1.0 - alpha) * loss_huber_batch + alpha * loss_margin_batch
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} done.")
    
    acc = evaluate_model_inter_branch(model, device)
    rho_boxes, rho_player = diagnose_spearman(model, device)
    print(f"Results for alpha = {alpha}:")
    print(f"  Optuna Inter-Branch Acc: {acc:.4f}")
    print(f"  Spearman (Fixed Player, Boxes Vary): {rho_boxes:.3f}")
    print(f"  Spearman (Fixed Boxes, Player Vary): {rho_player:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    run_pilot(args.alpha, args.margin, args.epochs)
