"""
PASO 2 — Entrenamiento del Clasificador Contrastivo v2 (Dataset Combinado).
Usa los hiperparámetros exactos de Trial 14 del enjambre Optuna.

PREREQUISITOS:
  1. Haber corrido verify_combined_dataset.py y confirmado el balance.
  2. Los tensores contrastive_fold_0_*.pt deben incluir datos densos (t==2/t==3).
  3. models/resnet.py debe contener SokobanSEResNetClassifier.

Uso:
  cd ~/hans/FI-Sokoban-Generator
  source venv/bin/activate
  python3 surrogate_models/train_production_contrastive_v2.py

Guarda el checkpoint como: production_contrastive_classifier_v2_combined.pt
NO sobrescribe el checkpoint anterior (production_contrastive_classifier.pt).
"""
import sys, os, json, random, hashlib, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

from models.resnet import SokobanSEResNetClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS FIJOS — Trial 14 del Enjambre Optuna v3
# Fuente: best_hparams_contrastive_classifier.json
# ─────────────────────────────────────────────────────────────────────────────
LR           = 0.00468
WEIGHT_DECAY = 0.000001
DROPOUT_P    = 0.10
BATCH_SIZE   = 512
EPOCHS       = 15
OPTIMAL_THRESHOLD = 0.70

OUTPUT_NAME = "production_contrastive_classifier_v2_combined.pt"

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class ProductionContrastiveDataset(Dataset):
    def __init__(self, X, y, t, is_train=True):
        self.X = X
        self.y = y
        self.t = t
        self.is_train = is_train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            # Aumentación D4 dinámica on-the-fly (12 canales en simultáneo)
            k = random.randint(0, 3)
            flip = random.choice([True, False])
            x = torch.rot90(x, k, [1, 2])
            if flip:
                x = torch.flip(x, [2])
        return x, self.y[idx], self.t[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    start_wall = time.time()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("  ENTRENAMIENTO — Clasificador Contrastivo v2 (Dataset Combinado)")
    print("=" * 70)
    print(f"  Inicio:       {start_ts}")
    print(f"  Dispositivo:  {device}")
    print(f"  Hiperparámetros (Trial 14 Optuna):")
    print(f"    LR={LR} | WD={WEIGHT_DECAY} | Dropout={DROPOUT_P} | BS={BATCH_SIZE}")
    print(f"    Épocas={EPOCHS} | Umbral calibrado={OPTIMAL_THRESHOLD}")
    print(f"  Checkpoint de salida: {OUTPUT_NAME}")
    print()

    # ── Cargar y unificar 100% del dataset ────────────────────────────────────
    fold_files = {
        "X_train": "contrastive_fold_0_X_train.pt",
        "y_train": "contrastive_fold_0_y_train.pt",
        "t_train": "contrastive_fold_0_t_train.pt",
        "X_test":  "contrastive_fold_0_X_test.pt",
        "y_test":  "contrastive_fold_0_y_test.pt",
        "t_test":  "contrastive_fold_0_t_test.pt",
    }

    for label, fname in fold_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"❌ No se encontró {fname}. Abortando.")
            return

    print("📦 Cargando tensores de fold 0 (train + test)...")
    X_train = torch.load(os.path.join(RESULTS_DIR, fold_files["X_train"]), map_location="cpu", weights_only=False)
    y_train = torch.load(os.path.join(RESULTS_DIR, fold_files["y_train"]), map_location="cpu", weights_only=False)
    t_train = torch.load(os.path.join(RESULTS_DIR, fold_files["t_train"]), map_location="cpu", weights_only=False)
    X_test  = torch.load(os.path.join(RESULTS_DIR, fold_files["X_test"]),  map_location="cpu", weights_only=False)
    y_test  = torch.load(os.path.join(RESULTS_DIR, fold_files["y_test"]),  map_location="cpu", weights_only=False)
    t_test  = torch.load(os.path.join(RESULTS_DIR, fold_files["t_test"]),  map_location="cpu", weights_only=False)

    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    t_all = torch.cat([t_train, t_test], dim=0)

    del X_train, y_train, t_train, X_test, y_test, t_test  # Liberar RAM

    total = len(y_all)
    num_pos = int((y_all == 1).sum())
    num_neg = int((y_all == 0).sum())
    num_t1 = int((t_all == 1).sum())
    num_t2 = int((t_all == 2).sum())
    num_t3 = int((t_all == 3).sum())
    pos_weight_val = num_neg / max(1, num_pos)

    print(f"\n📊 Dataset Unificado:")
    print(f"   Total: {total:,} | Pos: {num_pos:,} | Neg: {num_neg:,} | Ratio neg/pos: {pos_weight_val:.3f}")
    print(f"   t==1 (Solvable): {num_t1:,} | t==2 (Simple DL): {num_t2:,} | t==3 (Complex DL): {num_t3:,}")
    print(f"   Shape: {X_all.shape}")

    # Verificación de cordura: ¿hay datos densos?
    if num_t2 == 0 and num_t3 == 0:
        print("\n⚠️  ADVERTENCIA: No hay deadlocks densos (t==2/t==3) en el dataset.")
        print("    ¿Se regeneraron los tensores con datos densos? Abortando por seguridad.")
        return

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    dataset = ProductionContrastiveDataset(X_all, y_all, t_all, is_train=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SokobanSEResNetClassifier(dropout_p=DROPOUT_P, in_channels=12).to(device)
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n🚀 Iniciando entrenamiento ({EPOCHS} épocas, CosineAnnealingLR)...\n")
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch, _ in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - train_start
        print(f"   Epoch [{epoch:02d}/{EPOCHS}] | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Tiempo: {elapsed:.0f}s")

    train_elapsed = time.time() - train_start
    print(f"\n✅ Entrenamiento completado en {train_elapsed:.1f} segundos ({train_elapsed/60:.1f} min).")

    # ── Guardar checkpoint ────────────────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, OUTPUT_NAME)
    torch.save(model.state_dict(), output_path)
    sha256 = compute_sha256(output_path)
    size_mb = os.path.getsize(output_path) / 1e6

    print(f"\n💾 Checkpoint guardado:")
    print(f"   Archivo:  {output_path}")
    print(f"   Tamaño:   {size_mb:.1f} MB")
    print(f"   SHA-256:  {sha256}")

    # ── Evaluación de cordura (100% dataset, sin augmentación) ────────────────
    print(f"\n🔍 Evaluación de cordura sobre 100% del dataset (umbral={OPTIMAL_THRESHOLD:.2f})...")
    eval_ds = ProductionContrastiveDataset(X_all, y_all, t_all, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)

    model.eval()
    all_probs, all_targets, all_types = [], [], []
    with torch.no_grad():
        for X_b, y_b, t_b in eval_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(y_b.numpy().flatten())
            all_types.extend(t_b.numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_types = np.array(all_types)

    # Métricas por umbral
    print(f"\n  {'Umbral':<8} | {'F0.5':<8} | {'Prec':<8} | {'Rec':<8}")
    print(f"  {'-'*40}")
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        preds = (all_probs >= thresh).astype(float)
        f05 = fbeta_score(all_targets, preds, beta=0.5, zero_division=0)
        prec = precision_score(all_targets, preds, zero_division=0)
        rec = recall_score(all_targets, preds, zero_division=0)
        print(f"  {thresh:<8.2f} | {f05:<8.4f} | {prec:<8.4f} | {rec:<8.4f}")

    # Especificidad por tipo de deadlock
    for t_val, t_name in [(2, "Simple"), (3, "Complex")]:
        mask = all_types == t_val
        n = int(mask.sum())
        if n > 0:
            preds_sub = (all_probs[mask] >= OPTIMAL_THRESHOLD).astype(float)
            spec = float(np.mean(preds_sub == 0))
            print(f"\n  Especificidad ({t_name} Deadlock, t=={t_val}, N={n:,}): {spec:.4f}")

    # Percentiles
    pos_probs = all_probs[all_targets == 1]
    neg_probs = all_probs[all_targets == 0]
    print(f"\n  Distribución de Probabilidades:")
    print(f"    Positivos (n={len(pos_probs):,}): Media={np.mean(pos_probs):.4f} | p5={np.percentile(pos_probs,5):.4f} | p50={np.median(pos_probs):.4f} | p95={np.percentile(pos_probs,95):.4f}")
    print(f"    Negativos (n={len(neg_probs):,}): Media={np.mean(neg_probs):.4f} | p5={np.percentile(neg_probs,5):.4f} | p50={np.median(neg_probs):.4f} | p95={np.percentile(neg_probs,95):.4f}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_elapsed = time.time() - start_wall

    print(f"\n{'='*70}")
    print(f"  RESUMEN DE TRAZABILIDAD")
    print(f"{'='*70}")
    print(f"  Inicio:              {start_ts}")
    print(f"  Fin:                 {end_ts}")
    print(f"  Duración total:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Dataset:             {total:,} muestras (Original + Denso)")
    print(f"  Hiperparámetros:     Trial 14 (LR={LR}, WD={WEIGHT_DECAY}, Drop={DROPOUT_P}, BS={BATCH_SIZE})")
    print(f"  Checkpoint:          {OUTPUT_NAME}")
    print(f"  SHA-256:             {sha256}")
    print(f"  Dispositivo:         {device}")
    print(f"{'='*70}")
    print(f"\n🎯 Ahora corré el PASO 3: evaluate_contrastive_v2.py")

if __name__ == "__main__":
    main()
