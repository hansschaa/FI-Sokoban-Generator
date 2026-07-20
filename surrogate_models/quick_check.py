"""
quick_check.py
--------------
Sondeo inicial: entrena el Regresor solo en el Fold 1 durante 50 épocas.
Objetivo: verificar que el modelo aprende (MAE debe bajar de ~40 a ~10-15).

Ejecutar en tmux:
    venv/bin/python surrogate_models/quick_check.py
"""

import sys, os, copy, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

from models.resnet import SokobanResNetRegressor, MultiHeadRegressorLoss

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FOLD        = 1
EPOCHS      = 50
PATIENCE    = 12
BATCH_SIZE  = 128

# ─── DATASET ──────────────────────────────────────────────────────────────────
class FoldDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            d['tensor'].float(),
            torch.tensor(d['pushes_norm'], dtype=torch.float32),
            torch.tensor(d['branch_norm'], dtype=torch.float32),
            torch.tensor(d['pushes_raw'],  dtype=torch.float32),
            torch.tensor(d['branch_raw'],  dtype=torch.float32),
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  SONDEO INICIAL — Fold {FOLD} | {EPOCHS} épocas")
    print(f"  Dispositivo: {device.type.upper()}", end="")
    if device.type == "cuda":
        print(f"  →  {torch.cuda.get_device_name(0)}")
    else:
        print()
    print(f"{'='*55}\n")

    print("Cargando datos del Fold 1...")
    train_data = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_train.pt", weights_only=False)
    test_data  = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_test.pt",  weights_only=False)
    stats      = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_stats.pt", weights_only=False)
    p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]
    b_mean, b_std = stats["branch_mean"], stats["branch_std"]
    print(f"Train: {len(train_data):,} | Test: {len(test_data):,}")

    bucket_counts  = Counter(d["bucket"] for d in train_data)
    print("\nDistribución de buckets (train):")
    for k in sorted(bucket_counts): print(f"  {k:15s}: {bucket_counts[k]:>6,}")

    sample_weights = [1.0 / bucket_counts[d["bucket"]] for d in train_data]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(FoldDataset(train_data), batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(FoldDataset(test_data),  batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=True)

    model     = SokobanResNetRegressor(dropout_p=0.4).to(device)
    criterion = MultiHeadRegressorLoss(w_pushes=1.0, w_branch=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros del modelo: {n_params:,}")

    print(f"\n{'─'*55}")
    best_mae     = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for tensors, p_norm, b_norm, _, _ in train_loader:
            tensors = tensors.to(device)
            optimizer.zero_grad()
            p_pred, b_pred = model(tensors)
            loss, _ = criterion(p_pred, p_norm.to(device), b_pred, b_norm.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        total_mae_p, total_mae_b, n = 0.0, 0.0, 0
        with torch.no_grad():
            for tensors, _, _, p_raw, b_raw in test_loader:
                p_pred, b_pred = model(tensors.to(device))
                p_dn = p_pred.cpu() * p_std + p_mean
                b_dn = b_pred.cpu() * b_std + b_mean
                total_mae_p += torch.abs(p_dn - p_raw).sum().item()
                total_mae_b += torch.abs(b_dn - b_raw).sum().item()
                n += len(p_raw)

        mae_p = total_mae_p / n
        mae_b = total_mae_b / n
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(mae_p)
        lr_now = optimizer.param_groups[0]["lr"]

        tag = ""
        if mae_p < best_mae:
            best_mae     = mae_p
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = "  ★"
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n🛑 Early Stopping en época {epoch}.")
                break

        print(f"Época {epoch:03d}/{EPOCHS} | Loss: {avg_loss:.3f} | "
              f"MAE Pushes: {mae_p:.2f} | MAE Branch: {mae_b:.3f} | "
              f"LR: {lr_now:.1e}{tag}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*55}")
    print(f"  ✅ Sondeo completado en {elapsed:.1f} minutos")
    print(f"  Mejor MAE Pushes: {best_mae:.2f} empujes")
    print(f"{'='*55}")

    out = os.path.join(RESULTS_DIR, "quick_check_fold1.pt")
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), out)
    print(f"  Checkpoint guardado: {out}")

    print(f"\n  Diagnóstico:")
    if best_mae < 10:
        print("  🟢 Excelente — el modelo aprende muy bien. Listo para Optuna.")
    elif best_mae < 20:
        print("  🟡 Bien — el modelo aprende. Optuna puede mejorar los hiperparámetros.")
    elif best_mae < 35:
        print("  🟠 Moderado — el modelo aprende pero lento. Optuna es necesario.")
    else:
        print("  🔴 Bajo — el modelo no converge bien. Revisar arquitectura/datos.")

if __name__ == '__main__':
    main()
