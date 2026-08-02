import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.resnet import SokobanSEResNetRegressor

RESULTS_DIR = "surrogate_models/results"
STATS_PATH = os.path.join(RESULTS_DIR, "production_regressor_stats.pt")
PROD_MODEL_PATH = os.path.join(RESULTS_DIR, "production_regressor.pt")
TRAIN_PT = os.path.join(RESULTS_DIR, "siamese_ranknet_train.pt")
TEST_PT = os.path.join(RESULTS_DIR, "siamese_ranknet_test_heldout.pt")
OUT_MODEL_PATH = os.path.join(RESULTS_DIR, "siamese_ranknet_pilot.pt")

class SiameseRankingDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return (
            item["tensor_A"],
            torch.tensor(item["norm_A"], dtype=torch.float32),
            torch.tensor(item["raw_A"], dtype=torch.float32),
            item["bucket_A"],
            item["tensor_B"],
            torch.tensor(item["norm_B"], dtype=torch.float32),
            torch.tensor(item["raw_B"], dtype=torch.float32),
            item["bucket_B"]
        )

def evaluate_ranking_metrics(model, loader, device, pushes_mean, pushes_std):
    model.eval()
    all_raws_A = []
    all_preds_A_raw = []
    all_raws_B = []
    all_preds_B_raw = []
    
    correct_ordering = 0
    total_non_tie = 0
    
    hard_correct_ordering = 0
    hard_total_non_tie = 0
    
    hard_raws = []
    hard_preds = []

    with torch.no_grad():
        for tA, nA, rawA, bA, tB, nB, rawB, bB in loader:
            tA = tA.to(device)
            tB = tB.to(device)
            
            pA_norm = model(tA)
            pB_norm = model(tB)
            
            pA_raw = (pA_norm.cpu().numpy() * pushes_std) + pushes_mean
            pB_raw = (pB_norm.cpu().numpy() * pushes_std) + pushes_mean
            
            pA_raw = np.expm1(pA_raw)
            pB_raw = np.expm1(pB_raw)
            
            rA = rawA.numpy()
            rB = rawB.numpy()

            for i in range(len(rA)):
                all_raws_A.append(rA[i])
                all_preds_A_raw.append(pA_raw[i])
                all_raws_B.append(rB[i])
                all_preds_B_raw.append(pB_raw[i])
                
                if rA[i] != rB[i]:
                    total_non_tie += 1
                    true_diff = rA[i] - rB[i]
                    pred_diff = pA_raw[i] - pB_raw[i]
                    if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                        correct_ordering += 1
                        if rA[i] >= 50 or rB[i] >= 50:
                            hard_correct_ordering += 1
                    if rA[i] >= 50 or rB[i] >= 50:
                        hard_total_non_tie += 1
                
                if rA[i] >= 50:
                    hard_raws.append(rA[i])
                    hard_preds.append(pA_raw[i])
                if rB[i] >= 50:
                    hard_raws.append(rB[i])
                    hard_preds.append(pB_raw[i])

    all_raws = np.array(all_raws_A + all_raws_B)
    all_preds = np.array(all_preds_A_raw + all_preds_B_raw)
    
    mae = np.mean(np.abs(all_raws - all_preds))
    spearman_global, _ = spearmanr(all_raws, all_preds)
    acc_global = (correct_ordering / total_non_tie) * 100.0 if total_non_tie > 0 else 0.0
    
    spearman_hard, _ = spearmanr(hard_raws, hard_preds) if len(hard_raws) > 10 else (0.0, 0.0)
    acc_hard = (hard_correct_ordering / hard_total_non_tie) * 100.0 if hard_total_non_tie > 0 else 0.0

    return mae, spearman_global, acc_global, spearman_hard, acc_hard, len(hard_raws)//2

def main():
    parser = argparse.ArgumentParser(description="Siamese RankNet Pilot Training")
    parser.add_argument("--epochs", type=int, default=8, help="Number of pilot epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of MarginRankingLoss vs Huber")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch without loading production weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*90)
    print(" 🚀 ENTRENAMIENTO DE PILOTO: SIAMESE RANKNET INTRA-SHELL REGRESSOR")
    print(" Arquitectura Siamesa con MarginRankingLoss (margin=0.05 en espacio norm) + Huber Loss")
    print("="*90)
    print(f" Dispositivo : {device} | Epochs: {args.epochs} | LR: {args.lr} | Alpha: {args.alpha}")

    if not os.path.exists(TRAIN_PT) or not os.path.exists(TEST_PT):
        print("❌ Error: No se encontraron las particiones siamese_ranknet_*.pt. Ejecuta primero prepare_siamese_ranknet.py")
        return

    train_pairs = torch.load(TRAIN_PT, weights_only=False)
    test_pairs  = torch.load(TEST_PT, weights_only=False)
    stats       = torch.load(STATS_PATH, weights_only=False)
    pushes_mean = stats["pushes_mean"]
    pushes_std  = stats["pushes_std"]

    print(f" 📊 Datos Train/Val (Folds 2-5): {len(train_pairs):,} pares")
    print(f" 🧪 Datos Held-out (Fold 1)    : {len(test_pairs):,} pares (topologías inéditas)")

    train_loader = DataLoader(SiameseRankingDataset(train_pairs), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(SiameseRankingDataset(test_pairs),  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = SokobanSEResNetRegressor(dropout_p=0.4).to(device)
    if not args.from_scratch and os.path.exists(PROD_MODEL_PATH):
        print(" 💡 Cargando pesos iniciales desde production_regressor.pt (Fine-tuning)...")
        model.load_state_dict(torch.load(PROD_MODEL_PATH, map_location=device, weights_only=True))
    else:
        print(" ⚠️  Entrenando desde cero (--from_scratch activado o no hay production_regressor.pt)...")

    huber_loss = nn.HuberLoss()
    ranking_loss = nn.MarginRankingLoss(margin=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n  Evaluación inicial (Epoch 0 - Antes de RankNet)...")
    mae_0, sp_glob_0, acc_glob_0, sp_hard_0, acc_hard_0, hard_n = evaluate_ranking_metrics(model, test_loader, device, pushes_mean, pushes_std)
    print(f"  [Pre-Train] MAE: {mae_0:.2f} | Acc Global: {acc_glob_0:.1f}% | Spearman Global: {sp_glob_0:.3f}")
    print(f"  [Pre-Train Hard >=50] Acc: {acc_hard_0:.1f}% | Spearman: {sp_hard_0:.3f} (sobre {hard_n} pares difíciles)\n")

    best_sp_hard = sp_hard_0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_huber, total_rank = 0.0, 0.0, 0.0
        n_batches = 0

        for tA, nA, rA, bA, tB, nB, rB, bB in train_loader:
            tA, nA = tA.to(device), nA.to(device)
            tB, nB = tB.to(device), nB.to(device)
            rA, rB = rA.to(device), rB.to(device)

            optimizer.zero_grad()
            pA = model(tA)
            pB = model(tB)

            l_huber = (huber_loss(pA, nA) + huber_loss(pB, nB)) / 2.0
            
            target = torch.where(nA > nB, 1.0, -1.0).to(device)
            mask = (nA != nB)
            if mask.sum() > 0:
                l_rank = ranking_loss(pA[mask], pB[mask], target[mask])
            else:
                l_rank = torch.tensor(0.0, device=device)

            loss = (1.0 - args.alpha) * l_huber + args.alpha * l_rank
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_huber += l_huber.item()
            total_rank += l_rank.item()
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        mae, sp_glob, acc_glob, sp_hard, acc_hard, _ = evaluate_ranking_metrics(model, test_loader, device, pushes_mean, pushes_std)
        
        tag = ""
        if sp_hard > best_sp_hard:
            best_sp_hard = sp_hard
            best_weights = copy.deepcopy(model.state_dict())
            tag = "🌟"

        print(f"  Ep {epoch:02d}/{args.epochs} [{dt:.1f}s] | Loss: {total_loss/n_batches:.4f} (H:{total_huber/n_batches:.4f}, R:{total_rank/n_batches:.4f}) | "
              + f"Test MAE: {mae:.2f} | Acc: {acc_glob:.1f}% | Sp Glob: {sp_glob:.3f} | Sp Hard (>=50): {sp_hard:.3f} | Acc Hard: {acc_hard:.1f}% {tag}")

    model.load_state_dict(best_weights)
    torch.save(best_weights, OUT_MODEL_PATH)
    print("\n" + "="*90)
    print(f" 🎉 ENTRENAMIENTO DEL PILOTO SIAMÉS TERMINADO")
    print(f" Mejor Spearman en zona crítica (>=50 empujes): {best_sp_hard:.3f} (inicial fue {sp_hard_0:.3f})")
    print(f" 💾 Modelo guardado en: {OUT_MODEL_PATH}")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
