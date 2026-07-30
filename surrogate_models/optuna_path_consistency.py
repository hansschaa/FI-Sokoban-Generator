import os
import argparse
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.resnet import SokobanSEResNetRegressor
from prepare_path_consistency import encode_board, simulate_path
from train_final_path_consistency import PathConsistencyDataset
import glob
from evaluate_inter_branch import get_valid_children

def evaluate_model_inter_branch(model, device, n_pairs=500):
    model.eval()
    import pandas as pd
    
    TSV_FILE = "results/path_consistency_heldout.tsv"
    SOK_FILE = "../sok_files/benchmark_stratified_heldout.sok"
    
    if not os.path.exists(TSV_FILE):
        print(f"Generating TSV for evaluation from {SOK_FILE}...")
        import subprocess
        cmd = ["../build/batch_solver", SOK_FILE, "hungarian", TSV_FILE]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"Failed to generate TSV: {e}")
            return 0.0
            
    print(f"Reading TSV: {TSV_FILE}")
    df = pd.read_csv(TSV_FILE, sep='\t')
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Total rows in TSV: {len(df)}")
    
    board_map = {}
    sources = [SOK_FILE]
    for src in sources:
        if os.path.exists(src):
            with open(src, 'r') as f:
                lines = f.readlines()
            current_name = None
            current_board = []
            for line in lines:
                line = line.rstrip()
                if ' - pushes:' in line or ' - moves:' in line:
                    if current_name and current_board:
                        board_map[current_name] = '\n'.join(current_board)
                    current_name = line.split(' - ')[0].strip()
                    current_board = []
                elif line:
                    current_board.append(line)
            if current_name and current_board:
                board_map[current_name] = '\n'.join(current_board)
    
    total_pairs = 0
    correct_pairs = 0
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            if total_pairs >= n_pairs: break
            if row['Status'] != 'SOLVED': continue
            if row['LURD_Path'] == 'NONE': continue
            
            name = str(row['LevelName']).split(' - ')[0].strip()
            if name not in board_map: continue
            
            board_str = board_map[name]
            lurd = row['LURD_Path']
            states = simulate_path(board_str, lurd)
            
            for i in range(len(states) - 1):
                if total_pairs >= n_pairs: break
                s_curr, _ = states[i]
                s_opt, _ = states[i+1]
                
                children = get_valid_children(s_curr)
                sub_children = [c for c in children if c != s_opt]
                
                if not sub_children: continue
                
                t_opt = encode_board(s_opt)
                pred_opt = model(torch.tensor(t_opt).unsqueeze(0).to(device)).item()
                
                for s_sub in sub_children:
                    t_sub = encode_board(s_sub)
                    pred_sub = model(torch.tensor(t_sub).unsqueeze(0).to(device)).item()
                    
                    total_pairs += 1
                    if pred_opt < pred_sub: # We want optimal z-score to be smaller
                        correct_pairs += 1

    print(f"Evaluated pairs: {total_pairs}, Correct pairs: {correct_pairs}")
    if total_pairs > 0:
        return correct_pairs / total_pairs
    return 0.0


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Sugerir hiperparámetros
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    margin = trial.suggest_float("margin", 0.01, 0.2)
    
    # 2. Configurar modelo y datos
    model = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
    
    dataset = PathConsistencyDataset(1, augment=True)
    
    # Dataset ya está en RAM, num_workers>0 causa 'too many fds' con miles de tensores
    num_workers = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, drop_last=True)
                            
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    huber_loss = nn.HuberLoss(delta=1.0)
    ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    # 3. Entrenamiento corto (10 épocas)
    epochs = 10
    model.train()
    import time
    
    for epoch in range(epochs):
        t0 = time.time()
        for batch in dataloader:
            x_board = batch['tensor1'].to(device)
            p1_raw = batch['pushes1'].float().to(device)
            x_sibling = batch['tensor2'].to(device)
            
            p_mean = 2.45
            p_std = 1.05
            y_target = (torch.log1p(p1_raw) - p_mean) / p_std
            
            optimizer.zero_grad()
            
            pred_opt = model(x_board).squeeze(-1)
            pred_sub = model(x_sibling).squeeze(-1)
            
            loss_huber = huber_loss(pred_opt, y_target)
            
            # Queremos que pred_opt < pred_sub, así que y=-1
            target_rank = torch.full_like(pred_opt, -1)
            loss_rank = ranking_loss(pred_opt, pred_sub, target_rank)
            
            total_loss = loss_huber + alpha * loss_rank
            total_loss.backward()
            optimizer.step()
        
        print(f"  Epoch {epoch+1}/{epochs} finalizada en {time.time()-t0:.1f}s")
            
    # 4. Evaluación Inter-branch
    acc = evaluate_model_inter_branch(model, device, n_pairs=500)
    
    # Optuna maximiza o minimiza según study.direction. Queremos maximizar acc.
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-name', type=str, default='path_consistency_optuna')
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()
    
    db_url = os.environ.get("OPTUNA_DB_URL", "sqlite:///path_consistency.db")
    print(f"Connecting to {db_url}...")
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=db_url,
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.n_trials)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Inter-branch Acc): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
