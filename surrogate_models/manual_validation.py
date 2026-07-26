import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, precision_recall_curve, auc

from models.resnet import SokobanResNetClassifier, ClassifierLoss

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BATCH  = 1024
EPOCHS = 12

def main():
    print("=== EXAMEN PILOTO: CLASIFICADOR ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_path = os.path.join(RESULTS_DIR, "classifier_fold1_train.pt")
    test_path  = os.path.join(RESULTS_DIR, "classifier_fold1_test.pt")

    if not os.path.exists(train_path):
        print(f"❌ Error: No se encontró {train_path}")
        sys.exit(1)

    print("Cargando dataset...")
    raw_train = torch.load(train_path, weights_only=False)
    raw_test  = torch.load(test_path,  weights_only=False)

    X_train = raw_train["tensor"]
    y_train = raw_train["is_solvable"].float()
    X_test  = raw_test["tensor"]
    y_test  = raw_test["is_solvable"].float()
    torch.cuda.synchronize()

    N = len(X_train)
    print(f"Train: {N:,} | Test: {len(X_test):,}")

    N_pos = (y_train == 1.0).sum().item()
    N_neg = N - N_pos
    pos_weight = N_neg / N_pos if N_pos > 0 else 1.0
    print(f"pos_weight={pos_weight:.2f}  ({N_pos:,} solubles vs {N_neg:,} deadlocks)")

    model     = SokobanResNetClassifier().to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    use_amp   = device.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_batches  = (N + BATCH - 1) // BATCH
    PRINT_EVERY = 5   # imprime cada 5 batches (~cada pocos segundos)

    print(f"\nEntrenando {EPOCHS} épocas...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N, device='cpu')
        running_loss = torch.zeros(1, device=device)
        t0 = time.time()

        for i in range(n_batches):
            idx = perm[i * BATCH : (i + 1) * BATCH]
            xb  = X_train[idx].to(device)
            yb  = y_train[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(xb)
                loss   = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Acumular sin .item() para no sincronizar la GPU en cada paso
            running_loss += loss.detach() * len(yb)

            # Barra de progreso cada 10% de la época
            if (i + 1) % PRINT_EVERY == 0 or i == n_batches - 1:
                pct  = (i + 1) / n_batches * 100
                bar  = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
                loss_now = running_loss.item() / ((i + 1) * BATCH)
                elapsed  = time.time() - t0
                eta      = elapsed / (i + 1) * (n_batches - i - 1)
                print(f"\r  Ep {epoch:02d} [{bar}] {pct:5.1f}%  "
                      f"loss={loss_now:.4f}  {elapsed:.0f}s  ETA {eta:.0f}s     ", end="", flush=True)

        loss_mean = running_loss.item() / N
        t1 = time.time()
        
        # Evaluar validacion solo cada 3 épocas o en la última
        if epoch % 3 == 0 or epoch == EPOCHS:
            all_probs, all_targets = [], []
            with torch.no_grad():
                for i in range(0, len(X_test), BATCH):
                    xb = X_test[i : i + BATCH].to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_targets.append(y_test[i : i + BATCH].numpy())
            
            all_probs   = np.concatenate(all_probs).flatten()
            all_targets = np.concatenate(all_targets).flatten()
            
            preds = (all_probs > 0.5).astype(int)
            acc = accuracy_score(all_targets, preds)
            f1  = f1_score(all_targets, preds, zero_division=0)
            
            precisions, recalls, _ = precision_recall_curve(all_targets, all_probs)
            auc_pr = auc(recalls, precisions)
            
            print(f"\n  → {t1-t0:.1f}s | Loss={loss_mean:.4f} | Acc={acc:.3f} | F1={f1:.3f} | AUC-PR={auc_pr:.3f}\n")
        else:
            print(f"\n  → {t1-t0:.1f}s | Loss={loss_mean:.4f} | (Validación omitida para ganar velocidad)\n")

    print("✅ Piloto OK. Si Acc y F1 suben por época → el modelo está aprendiendo.")

if __name__ == "__main__":
    main()
