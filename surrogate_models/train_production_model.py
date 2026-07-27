"""
train_production_model.py
--------------------------
Entrena el modelo FINAL DE PRODUCCION usando TODOS los datos disponibles.

El 5-fold CV ya nos dio la estimación honesta de generalización.
Ahora reentrenamos con el 100% de los datos para maximizar el aprendizaje
antes de deployar el modelo en el sistema de generación de puzzles.

Estrategia anti-leakage:
  - Se cargan los 5 conjuntos de TEST (particiones disjuntas que cubren TODO el dataset
    exactamente una vez). NO se usan los conjuntos de train (tienen solapamiento entre folds).
  - Augmentación D4 (x8) aplicada on-the-fly SOLO durante training.

Criterio de parada:
  - Sin validation set (ya evaluado via CV). Entrenamos por N_EPOCHS fijos
    basado en el scheduler CosineAnnealingLR (LR decae a 0 en T_max épocas).
  - Para el clasificador: N_EPOCHS=50, T_max=50
  - Para el regresor:     N_EPOCHS=60, T_max=60

Uso:
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/train_production_model.py --model classifier
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/train_production_model.py --model regressor
    PYTHONPATH=surrogate_models ./venv/bin/python3 surrogate_models/train_production_model.py --model classifier --restart
"""

import sys, os, json, copy, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

from models.resnet import SokobanSEResNetRegressor, SokobanSEResNetClassifier, ClassifierLoss
from data.board_utils import augment_tensor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS     = 5

# ── Hiperparámetros de producción ─────────────────────────────────────────────
CLASSIFIER_EPOCHS = 50   # T_max del scheduler (LR llega a 0 en esta época)
REGRESSOR_EPOCHS  = 60

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierProductionDataset(Dataset):
    """Dataset de produccion: todos los datos con augmentacion D4 x8."""
    def __init__(self, tensor_all, label_all, augment=True):
        self.tensors  = tensor_all   # (N, 6, 25, 25)
        self.labels   = label_all    # (N,)
        self.augment  = augment
        # Precompute: con augmentacion, cada sample tiene 8 variantes
        self.aug_factor = 8 if augment else 1

    def __len__(self):
        return len(self.labels) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx  = idx // self.aug_factor
        aug_idx   = idx %  self.aug_factor
        tensor_np = self.tensors[orig_idx].numpy()  # (6, 25, 25)
        label     = self.labels[orig_idx].float()

        if self.augment:
            aug_variants = list(augment_tensor(tensor_np))  # lista de 8 arrays (6,25,25)
            tensor_out   = torch.from_numpy(aug_variants[aug_idx].copy()).float()
        else:
            tensor_out = torch.from_numpy(tensor_np).float()

        return tensor_out, label


class RegressorProductionDataset(Dataset):
    """Dataset de produccion para el regresor con augmentacion D4 x8."""
    def __init__(self, data_list, pushes_mean, pushes_std, augment=True):
        self.data       = data_list
        self.mean       = pushes_mean
        self.std        = pushes_std
        self.augment    = augment
        self.aug_factor = 8 if augment else 1

    def __len__(self):
        return len(self.data) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx // self.aug_factor
        aug_idx  = idx %  self.aug_factor
        item     = self.data[orig_idx]
        tensor_np = item["tensor"].numpy()
        pushes_norm = torch.tensor(item["pushes_norm"], dtype=torch.float32)
        pushes_raw  = torch.tensor(item["pushes_raw"],  dtype=torch.float32)
        weight      = torch.tensor(item.get("weight", 1.0), dtype=torch.float32)

        if self.augment:
            aug_variants = list(augment_tensor(tensor_np))
            tensor_out   = torch.from_numpy(aug_variants[aug_idx].copy()).float()
        else:
            tensor_out = torch.from_numpy(tensor_np).float()

        return tensor_out, pushes_norm, pushes_raw, weight


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def train_classifier_production(restart=False):
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams_classifier.json")
    if not os.path.exists(hparams_path):
        print("Error: No se encontro best_hparams_classifier.json en results/")
        return

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr         = cfg["lr"]
    wd         = cfg["weight_decay"]
    dropout_p  = cfg["dropout_p"]
    pos_weight = cfg["pos_weight"]
    batch_size = int(cfg["batch_size"])

    print("\n" + "="*65)
    print("  ENTRENAMIENTO PRODUCCION: CLASIFICADOR (SE-ResNet)")
    print("="*65)
    print(f"  Dispositivo   : {device}")
    print(f"  Hiper params  : lr={lr:.6f}, wd={wd:.6f}, drop={dropout_p:.4f}, pos_w={pos_weight:.2f}, bs={batch_size}")
    print(f"  Epocas fijas  : {CLASSIFIER_EPOCHS} (CosineAnnealingLR T_max={CLASSIFIER_EPOCHS})")
    print(f"  Datos         : 5 test-sets disjuntos (dataset completo, sin solapamiento)")
    print()

    # Cargar y concatenar los 5 test sets (particiones disjuntas = dataset completo)
    print("  Cargando los 5 test sets del clasificador...")
    all_tensors, all_labels = [], []
    for fold in range(1, N_FOLDS + 1):
        test_path = os.path.join(RESULTS_DIR, f"classifier_fold{fold}_test.pt")
        if not os.path.exists(test_path):
            print(f"  Warning: No existe {test_path}. Saltando.")
            continue
        data = torch.load(test_path, weights_only=False)
        all_tensors.append(data["tensor"])
        all_labels.append(data["is_solvable"])
        n_pos = int(data["is_solvable"].sum())
        n_tot = len(data["is_solvable"])
        print(f"    Fold {fold}: {n_tot:,} ejemplos ({n_pos:,} solubles, {n_tot-n_pos:,} deadlocks)")

    tensor_all = torch.cat(all_tensors, dim=0)
    label_all  = torch.cat(all_labels,  dim=0)
    n_total    = len(label_all)
    n_pos_total = int(label_all.sum())
    print(f"\n  Dataset total: {n_total:,} ejemplos ({n_pos_total:,} solubles, {n_total-n_pos_total:,} deadlocks)")
    print(f"  Con augmentacion D4 x8: {n_total*8:,} ejemplos de entrenamiento")
    real_ratio = (n_total - n_pos_total) / n_pos_total
    print(f"  Ratio real neg/pos: {real_ratio:.2f} | pos_weight Optuna: {pos_weight:.2f}")

    dataset    = ClassifierProductionDataset(tensor_all, label_all, augment=True)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model     = SokobanSEResNetClassifier(dropout_p=dropout_p).to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLASSIFIER_EPOCHS)

    ckpt_path = os.path.join(RESULTS_DIR, "production_classifier_ckpt.pt")
    out_path  = os.path.join(RESULTS_DIR, "production_classifier.pt")
    start_epoch = 1

    if restart and os.path.exists(ckpt_path):
        print(f"  -> Borrando checkpoint previo para empezar desde cero.")
        os.remove(ckpt_path)

    if os.path.exists(ckpt_path) and not restart:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  -> Reanudando desde epoca {start_epoch}")

    print()
    for epoch in range(start_epoch, CLASSIFIER_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        lr_now   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t0
        print(f"  Ep {epoch:02d}/{CLASSIFIER_EPOCHS} | T: {elapsed:.1f}s | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

        # Checkpoint cada 5 épocas o en la última
        if epoch % 5 == 0 or epoch == CLASSIFIER_EPOCHS:
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)

    # Guardar modelo de producción final
    torch.save(model.state_dict(), out_path)
    print(f"\n  Modelo de produccion guardado en: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────
def train_regressor_production(restart=False):
    hparams_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    if not os.path.exists(hparams_path):
        print("Error: No se encontro best_hparams.json en results/")
        return

    with open(hparams_path, "r") as f:
        cfg = json.load(f)["params"]

    lr         = cfg["lr"]
    wd         = cfg["weight_decay"]
    dropout_p  = cfg["dropout_p"]
    batch_size = int(cfg["batch_size"])

    print("\n" + "="*65)
    print("  ENTRENAMIENTO PRODUCCION: REGRESOR (SE-ResNet)")
    print("="*65)
    print(f"  Dispositivo  : {device}")
    print(f"  Hiper params : lr={lr:.6f}, wd={wd:.6f}, drop={dropout_p:.4f}, bs={batch_size}")
    print(f"  Epocas fijas : {REGRESSOR_EPOCHS} (CosineAnnealingLR T_max={REGRESSOR_EPOCHS})")
    print(f"  Datos        : 5 test-sets disjuntos (dataset completo, sin solapamiento)")
    print()

    print("  Cargando los 5 test sets del regresor...")
    all_data = []
    for fold in range(1, N_FOLDS + 1):
        test_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
        if not os.path.exists(test_path):
            print(f"  Warning: No existe {test_path}. Saltando.")
            continue
        data = torch.load(test_path, weights_only=False)
        all_data.extend(data)
        print(f"    Fold {fold}: {len(data):,} ejemplos")

    # Calcular estadísticas de normalización sobre TODO el dataset (sin leakage porque es producción)
    pushes_log   = np.log1p([item["pushes_raw"] for item in all_data])
    pushes_mean  = float(pushes_log.mean())
    pushes_std   = float(pushes_log.std())
    print(f"\n  Stats globales: pushes_log = {pushes_mean:.3f} +- {pushes_std:.3f}")

    # Re-normalizar con stats globales
    for item in all_data:
        item["pushes_norm"] = (np.log1p(item["pushes_raw"]) - pushes_mean) / (pushes_std + 1e-8)

    print(f"  Dataset total: {len(all_data):,} ejemplos")
    print(f"  Con augmentacion D4 x8: {len(all_data)*8:,} ejemplos de entrenamiento")

    dataset = RegressorProductionDataset(all_data, pushes_mean, pushes_std, augment=True)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model     = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
    criterion = nn.HuberLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=REGRESSOR_EPOCHS)

    ckpt_path = os.path.join(RESULTS_DIR, "production_regressor_ckpt.pt")
    out_path  = os.path.join(RESULTS_DIR, "production_regressor.pt")
    stats_path = os.path.join(RESULTS_DIR, "production_regressor_stats.pt")
    start_epoch = 1

    if restart and os.path.exists(ckpt_path):
        print(f"  -> Borrando checkpoint previo para empezar desde cero.")
        os.remove(ckpt_path)

    if os.path.exists(ckpt_path) and not restart:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  -> Reanudando desde epoca {start_epoch}")

    print()
    for epoch in range(start_epoch, REGRESSOR_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for x, y_norm, y_raw, w in loader:
            x, y_norm, w = x.to(device), y_norm.to(device), w.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = (criterion(pred, y_norm) * w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        lr_now   = scheduler.get_last_lr()[0]
        elapsed  = time.time() - t0
        print(f"  Ep {epoch:02d}/{REGRESSOR_EPOCHS} | T: {elapsed:.1f}s | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

        if epoch % 5 == 0 or epoch == REGRESSOR_EPOCHS:
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)

    # Guardar modelo y stats de normalización para inferencia
    torch.save(model.state_dict(), out_path)
    torch.save({"pushes_mean": pushes_mean, "pushes_std": pushes_std}, stats_path)
    print(f"\n  Modelo de produccion guardado en: {out_path}")
    print(f"  Stats de normalizacion guardados en: {stats_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo final de produccion con todos los datos.")
    parser.add_argument("--model",   choices=["classifier", "regressor", "both"], default="both",
                        help="Modelo a entrenar (default: both)")
    parser.add_argument("--restart", action="store_true",
                        help="Borrar checkpoint previo y empezar desde cero")
    args = parser.parse_args()

    if args.model in ("classifier", "both"):
        train_classifier_production(restart=args.restart)

    if args.model in ("regressor", "both"):
        train_regressor_production(restart=args.restart)
