"""
optuna_regressor.py
--------------------
Búsqueda de hiperparámetros con Optuna para el Surrogate Regressor.

- Usa SOLO el Fold 1 para ser eficiente (no multiplica por 5 el tiempo)
- Busca: lr, weight_decay, dropout_p, w_branch, batch_size
- Cada trial tiene Early Stopping propio (paciencia=10) y Pruning de Optuna
- Al terminar, guarda los mejores hiperparámetros en results/best_hparams.json

Ejecutar en tmux desde la raíz del proyecto:
    export OPTUNA_DB_URL="sqlite:///results/optuna.db"
    export OPTUNA_STUDY_NAME="sokoban_regressor"
    venv/bin/python surrogate_models/optuna_regressor.py

Duración estimada: 30 trials x ~12 min = ~6 horas (GPU de oficina)
"""

import sys, os, json, copy, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from collections import Counter
import optuna
from optuna.pruners import MedianPruner
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
from models.resnet import SokobanSEResNetRegressor
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
N_TRIALS    = 30
FOLD        = 1          # Solo Fold 1 para la búsqueda
MAX_EPOCHS  = 60         # Epochs reducidas por trial
PATIENCE    = 10

# Configurar storage distribuido si existe la variable de entorno
db_url = os.environ.get("OPTUNA_DB_URL", f"sqlite:///{RESULTS_DIR}/optuna_regressor.db")
study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_regressor")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*55}")
print(f"  OPTUNA — Surrogate Regressor (Fold {FOLD})")
print(f"  Dispositivo: {device.type.upper()}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Trials: {N_TRIALS} | Max épocas/trial: {MAX_EPOCHS}")
print(f"  Storage: {db_url}")
print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class FoldDataset(Dataset):
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

# Cargar datos UNA sola vez (no recargar en cada trial)
print(f"Cargando Fold {FOLD} (esto puede tardar ~30s)...")
_train_data = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_train.pt", weights_only=False)
_val_data   = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_val.pt",  weights_only=False)
_stats      = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_stats.pt", weights_only=False)
p_mean, p_std = _stats["pushes_mean"], _stats["pushes_std"]
b_mean, b_std = _stats["branch_mean"], _stats["branch_std"]
print(f"Train: {len(_train_data):,} | Validation: {len(_val_data):,}")
print(f"Stats — pushes: {p_mean:.1f}±{p_std:.1f} | branch: {b_mean:.2f}±{b_std:.2f}\n")

_val_dataset = FoldDataset(_val_data)


def make_loaders(batch_size):
    bucket_counts  = Counter(d["bucket"] for d in _train_data)
    sample_weights = [1.0 / bucket_counts[d["bucket"]] for d in _train_data]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(FoldDataset(_train_data), batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(_val_dataset, batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial):
    # Espacio de búsqueda
    lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_p    = trial.suggest_float("dropout_p",    0.1,  0.5)
    batch_size   = trial.suggest_categorical("batch_size", [128, 256])

    print(f"\n[Trial {trial.number}] -> Iniciando Trial...")
    print("  -> Creando Dataloaders...")
    train_loader, val_loader = make_loaders(batch_size)
    print("  -> Dataloaders creados. Inicializando modelo...")
    model     = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
    print("  -> Modelo en GPU. Configurando optimizador...")
    criterion = nn.HuberLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_mae    = float("inf")
    patience_ctr = 0

    print(f"  -> ¡Todo listo! Arrancando época 1...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        batch_idx = 0
        total_batches = len(train_loader)
        train_loss = 0.0
        for tensors, p_norm, _, weights in train_loader:
            if batch_idx == 0:
                print(f"    [Trial {trial.number}] Epoca {epoch:02d} | ¡Primer batch completado! La GPU está viva.")
            elif batch_idx % 10 == 0:
                print(f"    [Trial {trial.number}] Epoca {epoch:02d} | Progreso: {batch_idx}/{total_batches} batches...")

            tensors, p_norm, weights = tensors.to(device), p_norm.to(device), weights.to(device)

            optimizer.zero_grad()
            p_pred = model(tensors)
            
            loss_p = criterion(p_pred, p_norm)
            loss = (loss_p * weights).mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            batch_idx += 1

        train_loss /= len(train_loader)

        # ── Eval ─────────────────────────────────────────────────────────────
        model.eval()
        total_mae, n = 0.0, 0
        with torch.no_grad():
            for tensors, _, p_raw, _ in val_loader:
                tensors = tensors.to(device)
                p_pred = model(tensors)
                p_desnorm = p_pred.cpu() * p_std + p_mean
                p_desnorm_real = torch.expm1(p_desnorm)
                total_mae += torch.abs(p_desnorm_real - p_raw).sum().item()
                n += len(p_raw)
        mae = total_mae / n
        scheduler.step()

        print(f"  [Trial {trial.number}] Época {epoch:02d} | MAE Pushes: {mae:.2f}")

        # Pruning de Optuna (cancela trials malos antes de que terminen)
        trial.report(mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if mae < best_mae:
            best_mae     = mae
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_mae


# ─────────────────────────────────────────────────────────────────────────────
# ESTUDIO OPTUNA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=15),
        study_name=study_name,
        storage=optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600}
        ),
        load_if_exists=True,   # permite reanudar si se interrumpe
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*55)
    print("  RESULTADOS FINALES")
    print("="*55)
    print(f"  Mejor MAE Pushes: {study.best_value:.2f} empujes")
    print(f"  Mejores hiperparámetros:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Guardar en JSON para usarlos luego en el entrenamiento final
    out_path = os.path.join(RESULTS_DIR, "best_hparams.json")
    with open(out_path, "w") as f:
        json.dump({"best_mae": study.best_value, "params": study.best_params}, f, indent=2)
    print(f"\n  ✅ Hiperparámetros guardados en: {out_path}")

    # Resumen de todos los trials
    print(f"\n  Resumen de {len(study.trials)} trials:")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"    Completados: {len(completed)} | Podados: {len(pruned)}")
