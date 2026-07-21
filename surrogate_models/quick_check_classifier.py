import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
from sklearn.metrics import f1_score

from models.resnet import SokobanResNetClassifier, ClassifierLoss

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BATCH = 512
EPOCHS = 5

def main():
    print("=== EXAMEN PILOTO: CLASIFICADOR ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device.type.upper()}")

    train_path = os.path.join(RESULTS_DIR, "classifier_fold1_train.pt")
    test_path  = os.path.join(RESULTS_DIR, "classifier_fold1_test.pt")

    if not os.path.exists(train_path):
        print(f"❌ Error: No se encontró {train_path}")
        sys.exit(1)

    print("Cargando y moviendo dataset a GPU...")
    raw_train = torch.load(train_path, weights_only=False)
    raw_test  = torch.load(test_path,  weights_only=False)

    X_train = torch.stack([d["tensor"] for d in raw_train]).to(device)
    y_train = torch.tensor([d["is_solvable"] for d in raw_train], dtype=torch.float32).to(device)
    X_test  = torch.stack([d["tensor"] for d in raw_test]).to(device)
    y_test  = torch.tensor([d["is_solvable"] for d in raw_test], dtype=torch.float32).to(device)

    del raw_train, raw_test; gc.collect()

    N_train = len(X_train)
    print(f"Train (GPU): {N_train:,} | Test (GPU): {len(X_test):,}")

    N_pos = (y_train == 1.0).sum().item()
    N_neg = N_train - N_pos
    pos_weight = N_neg / N_pos if N_pos > 0 else 1.0
    print(f"pos_weight={pos_weight:.2f}")

    model     = SokobanResNetClassifier().to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    use_amp   = device.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\nEntrenando {EPOCHS} épocas (loop GPU puro, sin DataLoader)...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Shuffle manual en GPU — cero overhead de CPU
        perm = torch.randperm(N_train, device=device)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        train_loss = 0.0
        n_batches  = (N_train + BATCH - 1) // BATCH
        t0 = time.time()

        for i in range(n_batches):
            xb = X_shuf[i*BATCH : (i+1)*BATCH]
            yb = y_shuf[i*BATCH : (i+1)*BATCH]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(xb)
                loss   = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(yb)

            # Barra de progreso manual
            pct = (i+1) / n_batches
            bar = "█" * int(pct*40) + "░" * (40 - int(pct*40))
            print(f"\rEp {epoch:02d} Train [{bar}] {pct*100:5.1f}%  loss={loss.item():.4f}", end="", flush=True)

        train_loss /= N_train

        # Evaluación
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for i in range(0, len(X_test), BATCH):
                xb = X_test[i : i+BATCH]
                yb = y_test[i : i+BATCH]
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(xb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        f1  = f1_score(all_targets, all_preds, zero_division=0)
        elapsed = time.time() - t0

        print(f"\rEp {epoch:02d} | {elapsed:.1f}s | TrainLoss={train_loss:.4f} | Acc={acc:.3f} | F1={f1:.3f}   ")

    print("\n✅ Piloto finalizado. Si Acc y F1 suben → el clasificador está aprendiendo!")

if __name__ == "__main__":
    main()
