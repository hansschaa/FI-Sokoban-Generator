import os
import argparse
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.resnet import SokobanSEResNetRegressor
from prepare_path_consistency import PathConsistencyDataset, encode_board, parse_sok_files, simulate_path
from evaluate_inter_branch import get_valid_children

def evaluate_model_inter_branch(model, device, n_pairs=500):
    model.eval()
    import pandas as pd
    
    TSV_FILE = "../scratch/path_consistency_results.tsv"
    SOK_DIR = "../sokoban_dataset_buckets"
    
    if not os.path.exists(TSV_FILE):
        print("Warning: TSV file not found for evaluation.")
        return 0.0
        
    records = parse_sok_files(SOK_DIR, max_total=n_pairs)
    board_map = {r['name']: r['board_str'] for r in records}
    df = pd.read_csv(TSV_FILE, sep='\t')
    
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
    
    train_data = "results/path_consistency/path_fold1_train.pt"
    if not os.path.exists(train_data):
        raise FileNotFoundError(f"Missing {train_data}")
        
    dataset = PathConsistencyDataset(train_data, augment=True)
    
    # Check si podemos usar los workers
    num_workers = 4 if torch.cuda.is_available() else 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, drop_last=True)
                            
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    huber_loss = nn.HuberLoss(delta=1.0)
    ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    # 3. Entrenamiento corto (10 épocas)
    epochs = 10
    model.train()
    
    for epoch in range(epochs):
        for batch in dataloader:
            x_board = batch['board'].to(device)
            y_target = batch['target'].to(device)
            x_sibling = batch['sibling_board'].to(device)
            
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
