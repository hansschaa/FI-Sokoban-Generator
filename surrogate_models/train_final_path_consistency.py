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

def apply_d4(t, variant):
    if variant == 0: return t
    if variant == 1: return torch.flip(t, dims=[1])
    if variant == 2: return torch.flip(t, dims=[2])
    if variant == 3: return torch.flip(t, dims=[1, 2])
    
    t_T = torch.transpose(t, 1, 2)
    if variant == 4: return t_T
    if variant == 5: return torch.flip(t_T, dims=[1])
    if variant == 6: return torch.flip(t_T, dims=[2])
    if variant == 7: return torch.flip(t_T, dims=[1, 2])
    return t

class PathConsistencyDataset(Dataset):
    def __init__(self, fold_k, augment=True, max_route_distance=-1):
        self.augment = augment
        path_file = os.path.join(RESULTS_DIR, "path_consistency", f"path_fold{fold_k}_train.pt")
        print(f"  Cargando dataset de Path Consistency: {os.path.basename(path_file)}...")
        self.pairs = torch.load(path_file, weights_only=False, map_location='cpu')
        
        # Filtrar por distancia en la ruta (K=4 pushes por paso)
        if max_route_distance > 0:
            original_len = len(self.pairs)
            max_diff = max_route_distance * 4
            self.pairs = [p for p in self.pairs if (p['pushes1'] - p['pushes2']) <= max_diff]
            print(f"  Filtro max_route_distance={max_route_distance} aplicado. Pares: {original_len} -> {len(self.pairs)}")
        else:
            print(f"  Pares cargados: {len(self.pairs)}")
            
        # Cargar los pesos originales del fold de train (opcional — weight=1.0 si no existe)
        train_file = os.path.join(RESULTS_DIR, f"regressor_fold{fold_k}_train.pt")
        self.weight_map = {}
        if os.path.exists(train_file):
            print(f"  Extrayendo pesos originales de {os.path.basename(train_file)}...")
            orig_train = torch.load(train_file, weights_only=False, map_location='cpu')
            for item in orig_train:
                self.weight_map[item['shell_hash']] = item.get('weight', 1.0)
        else:
            print(f"  ⚠️  {os.path.basename(train_file)} no encontrado — usando weight=1.0 para todos los pares.")

    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        item = self.pairs[idx]
        t1 = item["tensor1"]
        t2 = item["tensor2"]
        
        # Convert numpy arrays to torch tensors if they aren't already
        if isinstance(t1, np.ndarray):
            t1 = torch.from_numpy(t1).float()
        if isinstance(t2, np.ndarray):
            t2 = torch.from_numpy(t2).float()
            
        p1 = item["pushes1"]
        p2 = item["pushes2"]
        shell_hash = item.get("shell_hash", "")
        weight = self.weight_map.get(shell_hash, 1.0)
        
        if self.augment:
            variant = torch.randint(0, 8, (1,)).item()
            t1 = apply_d4(t1, variant)
            t2 = apply_d4(t2, variant)
            
        return {
            "tensor1": t1.float(),
            "pushes1": p1,
            "tensor2": t2.float(),
            "pushes2": p2,
            "weight": weight
        }

class RegressorDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item['tensor'].float(),
            torch.tensor(item['pushes_norm'], dtype=torch.float32),
            torch.tensor(item['pushes_raw'],  dtype=torch.float32),
            torch.tensor(item.get('weight', 1.0), dtype=torch.float32),
        )

def train_path_consistency(folds_to_run, max_epochs, alpha, margin, max_route_distance, restart=False):
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams_path_consistency.json")
    if not os.path.exists(hparams_path):
        print(f"Error: No se encontró el archivo de hiperparámetros {hparams_path}")
        return

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr           = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    dropout_p    = cfg["dropout_p"]
    batch_size   = int(cfg["batch_size"]) // 2  # Usamos pares, reducimos el batch por memoria
    alpha        = cfg["alpha"]
    margin       = cfg["margin"]

    print("\n" + "="*65)
    print("  ENTRENAMIENTO FINAL: PATH CONSISTENCY REGRESSOR")
    print("="*65)
    print(f"  Dispositivo  : {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Hiperparámetros: lr={lr:.6f}, wd={weight_decay:.6f}, drop={dropout_p:.2f}, bs={batch_size} (ajustado por pares)")
    print(f"  Loss Params  : α={alpha} (peso del MarginRanking), margin={margin}\n")

    fold_maes = []

    for fold in folds_to_run:
        val_path   = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_val.pt")
        test_path  = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
        stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")

        if not os.path.exists(val_path):
            print(f"⚠️ Saltando Fold {fold}: No existe {val_path}")
            continue

        print(f"\n[{'─'*40}]")
        print(f"  INICIANDO FOLD {fold}/5")
        print(f"[{'─'*40}]")

        val_data   = torch.load(val_path,   weights_only=False)
        test_data  = torch.load(test_path,  weights_only=False)
        stats      = torch.load(stats_path, weights_only=False)
        p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]

        # Loader de pares de consistencia para el Train
        train_dataset = PathConsistencyDataset(fold, augment=True, max_route_distance=max_route_distance)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        # Loaders estándar para Val/Test
        val_loader   = DataLoader(RegressorDataset(val_data),   batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(RegressorDataset(test_data),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

        model     = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
        huber_loss = nn.HuberLoss(reduction='none')
        margin_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_mae     = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        patience_ctr = 0

        ckpt_path = os.path.join(RESULTS_DIR, "path_consistency", f"ckpt_regressor_fold{fold}.pt")
        start_epoch = 1
        
        if restart and os.path.exists(ckpt_path):
            print(f"  -> ⚠️ Bandera --restart detectada. Borrando checkpoint anterior para empezar desde cero.")
            os.remove(ckpt_path)
            
        if os.path.exists(ckpt_path):
            print(f"  -> 🔄 Reanudando desde checkpoint: {os.path.basename(ckpt_path)}")
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mae = ckpt['best_mae']
            best_weights = ckpt['best_weights']
            patience_ctr = ckpt['patience_ctr']

        for epoch in range(start_epoch, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            total_huber = 0.0
            total_margin = 0.0

            for batch in train_loader:
                t1 = batch["tensor1"].to(device)
                t2 = batch["tensor2"].to(device)
                p1_raw = batch["pushes1"].float().to(device)
                p2_raw = batch["pushes2"].float().to(device)
                weights = batch["weight"].float().to(device)
                
                optimizer.zero_grad()
                
                # Normalizar p_raw
                p1_norm = (torch.log1p(p1_raw) - p_mean) / p_std
                p2_norm = (torch.log1p(p2_raw) - p_mean) / p_std
                
                # IMPORTANTE: concatenar t1 y t2 en un solo batch para que BatchNorm
                # vea ambas distribuciones en la misma pasada y sus running_stats
                # no oscilen entre las dos sub-distribuciones separadas.
                combined = torch.cat([t1, t2], dim=0)
                pred_combined = model(combined).squeeze(-1)
                pred1, pred2 = pred_combined.split(t1.size(0))
                
                # Calcular HuberLoss ponderada por el peso del tablero original
                loss_huber1 = huber_loss(pred1, p1_norm)
                loss_huber2 = huber_loss(pred2, p2_norm)
                loss_huber_batch = ((loss_huber1 + loss_huber2) / 2.0 * weights).mean()
                
                # Calcular MarginRankingLoss
                # pred1 corresponde a p1, pred2 a p2. Como p1 > p2 siempre, queremos pred1 > pred2.
                # y = 1 significa que esperamos pred1 > pred2.
                y = torch.ones_like(pred1)
                loss_margin_batch = (margin_loss(pred1, pred2, y) * weights).mean()
                
                loss = (1.0 - alpha) * loss_huber_batch + alpha * loss_margin_batch
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                train_loss += loss.item()
                total_huber += loss_huber_batch.item()
                total_margin += loss_margin_batch.item()

            train_loss /= len(train_loader)
            total_huber /= len(train_loader)
            total_margin /= len(train_loader)

            model.eval()
            total_mae_val, n_val = 0.0, 0
            all_p_pred = []
            all_p_raw = []
            with torch.no_grad():
                for tensors, _, p_raw, _ in val_loader:
                    tensors = tensors.to(device)
                    p_pred = model(tensors)
                    p_desnorm = p_pred.cpu() * p_std + p_mean
                    p_desnorm_real = torch.expm1(p_desnorm)
                    
                    if p_raw.dim() == 1 and p_desnorm_real.dim() == 2:
                        p_desnorm_real = p_desnorm_real.squeeze(-1)
                    elif p_raw.dim() == 2 and p_desnorm_real.dim() == 1:
                        p_raw = p_raw.squeeze(-1)

                    total_mae_val += torch.abs(p_desnorm_real - p_raw).sum().item()
                    n_val += len(p_raw)
                    all_p_pred.extend(p_desnorm_real.view(-1).numpy())
                    all_p_raw.extend(p_raw.view(-1).numpy())

            val_mae = total_mae_val / n_val
            val_spearman, _ = spearmanr(all_p_raw, all_p_pred)
            scheduler.step()

            elapsed = time.time() - t0
            tag = ""
            if val_mae < best_mae:
                best_mae = val_mae
                best_weights = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                tag = " ★ (Nuevo récord Val)"
            else:
                patience_ctr += 1

            print(f"  Ep {epoch:02d} | T: {elapsed:.1f}s | Loss: {train_loss:.4f} (Huber: {total_huber:.4f}, Margin: {total_margin:.4f}) | MAE Val: {val_mae:.2f} empujes | Spearman Val: {val_spearman:.3f}{tag}")

            if patience_ctr >= 15:
                print(f"  🛑 Early Stopping en época {epoch}.")
                break
                
            # Guardar checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'best_weights': best_weights,
                'patience_ctr': patience_ctr
            }, ckpt_path)

        # ── Test Ciego ──
        model.load_state_dict(best_weights)
        model.eval()
        total_mae_test, n_test = 0.0, 0
        all_p_pred_test = []
        all_p_raw_test = []
        with torch.no_grad():
            for tensors, _, p_raw, _ in test_loader:
                tensors = tensors.to(device)
                p_pred = model(tensors)
                p_desnorm = p_pred.cpu() * p_std + p_mean
                p_desnorm_real = torch.expm1(p_desnorm)
                
                if p_raw.dim() == 1 and p_desnorm_real.dim() == 2:
                    p_desnorm_real = p_desnorm_real.squeeze(-1)
                elif p_raw.dim() == 2 and p_desnorm_real.dim() == 1:
                    p_raw = p_raw.squeeze(-1)
                    
                total_mae_test += torch.abs(p_desnorm_real - p_raw).sum().item()
                n_test += len(p_raw)
                all_p_pred_test.extend(p_desnorm_real.view(-1).numpy())
                all_p_raw_test.extend(p_raw.view(-1).numpy())
        
        test_mae = total_mae_test / n_test
        test_spearman, _ = spearmanr(all_p_raw_test, all_p_pred_test)
        fold_maes.append(test_mae)

        save_path = os.path.join(RESULTS_DIR, "path_consistency", f"final_regressor_fold{fold}.pt")
        torch.save(best_weights, save_path)
        print(f"  ✅ Fold {fold} guardado en {os.path.basename(save_path)} | MAE Val: {best_mae:.2f} | MAE TEST: {test_mae:.2f} | Spearman TEST: {test_spearman:.3f}")
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    if len(fold_maes) > 1:
        print(f"\n  🏆 REGRESOR FINAL (5-FOLD CV): MAE Pushes = {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} empujes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Final con Path Consistency y D4")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="Folds a ejecutar separados por coma (ej. 1)")
    parser.add_argument("--epochs", type=int, default=50, help="Máximo de épocas")
    parser.add_argument("--alpha", type=float, default=0.1, help="Peso del Margin Ranking Loss (0-1). Loss = (1-alpha)*Huber + alpha*Margin")
    parser.add_argument("--margin", type=float, default=0.05, help="Margen para el Margin Ranking Loss (en espacio z-score de log(pushes)). K=4 pasos -> diff real ~0.12; usar 0.05 como conservador.")
    parser.add_argument("--max_route_distance", type=int, default=1, help="Máxima distancia (en pasos K=4) entre estados del par. 1 = consecutivos. -1 = todos.")
    parser.add_argument("--restart", action="store_true", help="Ignora checkpoints existentes y reinicia el entrenamiento desde cero")

    args = parser.parse_args()
    folds_to_run = [int(x.strip()) for x in args.folds.split(",")]

    train_path_consistency(folds_to_run, args.epochs, args.alpha, args.margin, args.max_route_distance, restart=args.restart)
