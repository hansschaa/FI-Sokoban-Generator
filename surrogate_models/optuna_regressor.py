"""
optuna_regressor.py
--------------------
Búsqueda de hiperparámetros con Optuna para el Surrogate Regressor.

- Usa SOLO el Fold 1 para ser eficiente (no multiplica por 5 el tiempo)
- Busca: lr, weight_decay, dropout_p, w_branch, batch_size
- Cada trial tiene Early Stopping propio (paciencia=10) y Pruning de Optuna
- Al terminar, guarda los mejores hiperparámetros en results/best_hparams.json

Ejecutar en tmux desde la raíz del proyecto:
    venv/bin/python surrogate_models/optuna_regressor.py

Duración estimada: 30 trials x ~12 min = ~6 horas (GPU de oficina)
"""

import sys, os, json, copy, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import optuna
from optuna.pruners import MedianPruner

from models.resnet import SokobanResNetRegressor, MultiHeadRegressorLoss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
N_TRIALS    = 30
FOLD        = 1          # Solo Fold 1 para la búsqueda
MAX_EPOCHS  = 60         # Epochs reducidas por trial
PATIENCE    = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*55}")
print(f"  OPTUNA — Surrogate Regressor (Fold {FOLD})")
print(f"  Dispositivo: {device.type.upper()}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Trials: {N_TRIALS} | Max épocas/trial: {MAX_EPOCHS}")
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
            torch.tensor(item['pushes_norm'],  dtype=torch.float32),
            torch.tensor(item['branch_norm'],  dtype=torch.float32),
            torch.tensor(item['pushes_raw'],   dtype=torch.float32),
            torch.tensor(item['branch_raw'],   dtype=torch.float32),
        )

# Cargar datos UNA sola vez (no recargar en cada trial)
print(f"Cargando Fold {FOLD} (esto puede tardar ~30s)...")
_train_data = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_train.pt", weights_only=False)
_test_data  = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_test.pt",  weights_only=False)
_stats      = torch.load(f"{RESULTS_DIR}/regressor_fold{FOLD}_stats.pt", weights_only=False)
p_mean, p_std = _stats["pushes_mean"], _stats["pushes_std"]
b_mean, b_std = _stats["branch_mean"], _stats["branch_std"]
print(f"Train: {len(_train_data):,} | Test: {len(_test_data):,}")
print(f"Stats — pushes: {p_mean:.1f}±{p_std:.1f} | branch: {b_mean:.2f}±{b_std:.2f}\n")

_test_dataset = FoldDataset(_test_data)


def make_loaders(batch_size):
    bucket_counts  = Counter(d["bucket"] for d in _train_data)
    sample_weights = [1.0 / bucket_counts[d["bucket"]] for d in _train_data]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(FoldDataset(_train_data), batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(_test_dataset, batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial):
    # Espacio de búsqueda
    lr           = trial.suggest_float("lr",           1e-4, 5e-3,  log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3,  log=True)
    dropout_p    = trial.suggest_float("dropout_p",    0.2,  0.6)
    w_branch     = trial.suggest_float("w_branch",     0.1,  1.0)
    batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256])

    print(f"\n[Trial {trial.number}] -> Iniciando Trial...")
    print("  -> Creando Dataloaders...")
    train_loader, test_loader = make_loaders(batch_size)
    print("  -> Dataloaders creados. Inicializando modelo...")
    model     = SokobanResNetRegressor(dropout_p=dropout_p).to(device)
    print("  -> Modelo en GPU. Configurando optimizador...")
    criterion = MultiHeadRegressorLoss(w_pushes=1.0, w_branch=w_branch)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=4)

    best_mae    = float("inf")
    patience_ctr = 0

    print(f"  -> ¡Todo listo! Arrancando época 1...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        batch_idx = 0
        total_batches = len(train_loader)
        
        for tensors, p_norm, b_norm, _, _ in train_loader:
            if batch_idx > 0 and batch_idx % 400 == 0:
                print(f"    [Trial {trial.number}] Epoca {epoch:02d} | Progreso: {batch_idx}/{total_batches} batches...")
            
            tensors = tensors.to(device)
            p_norm  = p_norm.to(device)
            b_norm  = b_norm.to(device)
            optimizer.zero_grad()
            p_pred, b_pred = model(tensors)
            loss, _ = criterion(p_pred, p_norm, b_pred, b_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            batch_idx += 1

        # ── Eval ─────────────────────────────────────────────────────────────
        model.eval()
        total_mae, n = 0.0, 0
        with torch.no_grad():
            for tensors, _, _, p_raw, _ in test_loader:
                tensors = tensors.to(device)
                p_pred, _ = model(tensors)
                p_desnorm = p_pred.cpu() * p_std + p_mean
                total_mae += torch.abs(p_desnorm - p_raw).sum().item()
                n += len(p_raw)
        mae = total_mae / n
        scheduler.step(mae)

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
        study_name="sokoban_regressor",
        storage=f"sqlite:///{RESULTS_DIR}/optuna_regressor.db",
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
