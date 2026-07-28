"""
train_residual.py
-----------------
Entrena el regresor para predecir el *Residual* (pushes_raw - hungarian_lb)
en lugar de predecir pushes directamente.

Se asume que los tensores ya fueron procesados con `augment_hungarian.py`.
"""

import sys, os, json, copy, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import spearmanr

from models.resnet import SokobanSEResNetRegressor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegressorDatasetResidual(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['tensor'].float(),
            torch.tensor(item['residual_norm'], dtype=torch.float32),
            torch.tensor(item['residual_raw'],  dtype=torch.float32),
            torch.tensor(item['hungarian_lb'],  dtype=torch.float32),
            torch.tensor(item['pushes_raw'],    dtype=torch.float32),
            torch.tensor(item.get('weight', 1.0), dtype=torch.float32),
            item['bucket']
        )

def eval_spearman_by_bucket(p_raw_list, p_pred_list, bucket_list):
    from collections import defaultdict
    groups = defaultdict(lambda: ([], []))
    for raw, pred, b in zip(p_raw_list, p_pred_list, bucket_list):
        groups[b][0].append(raw)
        groups[b][1].append(pred)

    result = {}
    global_rho, _ = spearmanr(p_raw_list, p_pred_list)
    result["global"] = global_rho

    def bucket_sort_key(b):
        if b == '101_plus': return 101
        return int(b.split('_')[0])

    for b in sorted(groups.keys(), key=bucket_sort_key):
        raws, preds = groups[b]
        if len(raws) < 2:
            result[b] = float('nan')
        else:
            rho, _ = spearmanr(raws, preds)
            result[b] = rho
    return result

def train_residual_regressor(folds_to_run, max_epochs, restart):
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    if not os.path.exists(hparams_path):
        print("❌ Error: No se encontró best_hparams.json")
        return

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr           = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p    = cfg["dropout_p"]
    batch_size   = int(cfg["batch_size"])

    criterion = nn.HuberLoss(reduction='none')

    print("\n" + "="*70)
    print("  EXPERIMENT: RESIDUAL LEARNING")
    print("="*70)
    print(f"  Target        : residual_norm (Z-score de log1p(pushes - hungarian_lb))")
    print(f"  Dispositivo   : {device.type.upper()}")
    print(f"  lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, bs={batch_size}\n")

    fold_maes = []

    for fold in folds_to_run:
        train_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_train.pt")
        val_path   = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_val.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")

        if not os.path.exists(train_path):
            continue

        train_data = torch.load(train_path, weights_only=False)
        val_data   = torch.load(val_path,   weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)
        stats      = torch.load(stats_path, weights_only=False)
        
        # Validar que ya pasaron por augment_hungarian.py
        if 'residual_mean' not in stats:
            print(f"❌ Error en Fold {fold}: No se encontró 'residual_mean' en stats.pt. Corre augment_hungarian.py primero.")
            return

        r_mean, r_std = stats["residual_mean"], stats["residual_std"]

        train_loader = DataLoader(RegressorDatasetResidual(train_data), batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(RegressorDatasetResidual(val_data),   batch_size=256, shuffle=False, pin_memory=True)
        test_loader  = DataLoader(RegressorDatasetResidual(test_data),  batch_size=256, shuffle=False, pin_memory=True)

        model = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_mae     = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_residual_fold{fold}.pt")
        start_epoch = 1

        if restart and os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mae = ckpt['best_mae']
            best_weights = ckpt['best_weights']
            patience_ctr = ckpt['patience_ctr']

        print(f"\n[{'─'*50}]")
        print(f"  INICIANDO FOLD {fold}/5 (Residual)")
        print(f"[{'─'*50}]")

        for epoch in range(start_epoch, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0

            for tensors, r_norm, r_raw, h_lb, pushes_raw, weights, _ in train_loader:
                tensors = tensors.to(device)
                r_norm  = r_norm.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()
                r_pred = model(tensors)

                loss = (criterion(r_pred, r_norm) * weights).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            total_mae_val, n_val = 0.0, 0
            all_p_pred_val, all_p_raw_val, all_buckets_val = [], [], []
            
            with torch.no_grad():
                for tensors, _, r_raw, h_lb, pushes_raw, _, buckets in val_loader:
                    tensors = tensors.to(device)
                    r_pred = model(tensors)
                    
                    # Reconstruir pushes: e^(r_pred * std + mean) - 1 + hungarian_lb
                    r_desnorm = r_pred.cpu() * r_std + r_mean
                    r_desnorm_real = torch.expm1(r_desnorm)
                    p_pred_real = r_desnorm_real + h_lb
                    
                    total_mae_val += torch.abs(p_pred_real - pushes_raw).sum().item()
                    n_val += len(pushes_raw)
                    
                    all_p_pred_val.extend(p_pred_real.view(-1).tolist())
                    all_p_raw_val.extend(pushes_raw.view(-1).tolist())
                    all_buckets_val.extend(list(buckets))

            val_mae = total_mae_val / n_val
            sp_by_bucket = eval_spearman_by_bucket(all_p_raw_val, all_p_pred_val, all_buckets_val)
            val_sp_global = sp_by_bucket.get("global", float('nan'))
            val_sp_91 = np.nanmean([sp_by_bucket.get(b, float('nan')) for b in ('91_to_100', '101_plus')])

            scheduler.step()
            
            tag = " ★" if val_mae < best_mae else ""
            if val_mae < best_mae:
                best_mae = val_mae
                best_weights = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            print(f"  Ep {epoch:02d} | {time.time()-t0:.1f}s | Loss {train_loss:.4f} | MAE Pushes {val_mae:.2f} | Sp_global {val_sp_global:.3f} | Sp_91+ {val_sp_91:.3f}{tag}")

            if patience_ctr >= 15:
                print(f"  🛑 Early Stopping en época {epoch}.")
                break

            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae, 'best_weights': best_weights, 'patience_ctr': patience_ctr,
            }, ckpt_path)

        # ── Test Ciego ─────────────────────────────────────────────────────
        model.load_state_dict(best_weights)
        model.eval()
        total_mae_test, n_test = 0.0, 0
        all_p_pred_test, all_p_raw_test, all_buckets_test = [], [], []
        
        with torch.no_grad():
            for tensors, _, r_raw, h_lb, pushes_raw, _, buckets in test_loader:
                tensors = tensors.to(device)
                r_pred = model(tensors)
                r_desnorm = r_pred.cpu() * r_std + r_mean
                r_desnorm_real = torch.expm1(r_desnorm)
                p_pred_real = r_desnorm_real + h_lb
                
                total_mae_test += torch.abs(p_pred_real - pushes_raw).sum().item()
                n_test += len(pushes_raw)
                all_p_pred_test.extend(p_pred_real.view(-1).tolist())
                all_p_raw_test.extend(pushes_raw.view(-1).tolist())
                all_buckets_test.extend(list(buckets))

        test_mae = total_mae_test / n_test
        test_sp_by_bucket = eval_spearman_by_bucket(all_p_raw_test, all_p_pred_test, all_buckets_test)
        
        fold_maes.append(test_mae)
        save_path = os.path.join(RESULTS_DIR, f"residual_regressor_fold{fold}.pt")
        torch.save(best_weights, save_path)

        print(f"\n  ✅ Fold {fold} MAE TEST: {test_mae:.2f}")
        for b in sorted(test_sp_by_bucket.keys()):
            if b != 'global':
                print(f"       {b:<15}: {test_sp_by_bucket[b]:.3f}")

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    if len(fold_maes) > 1:
        print(f"\n  🏆 RESIDUAL REGRESSOR (5-FOLD CV): MAE = {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} empujes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    train_residual_regressor([int(x) for x in args.folds.split(",")], args.epochs, args.restart)
