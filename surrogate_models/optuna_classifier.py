"""
optuna_classifier.py
--------------------
Búsqueda de hiperparámetros con Optuna para el Surrogate Classifier.

- Usa SOLO el Fold 1 para ser eficiente
- Busca: lr, weight_decay, dropout_p, pos_weight, batch_size
- Métrica: F_0.5 (premia la Precisión sobre el Recall para evitar falsos positivos)
- Cada trial tiene Early Stopping propio y Pruning de Optuna
- Al terminar, guarda los mejores hiperparámetros en results/best_hparams_classifier.json

Ejecutar en los PCs del laboratorio (todos conectados a la misma DB):
    export OPTUNA_DB_URL="mysql+pymysql://USER:PASSWORD@HOST/optuna_db"
    export OPTUNA_STUDY_NAME="sokoban_classifier_lab_v4"
    venv/bin/python surrogate_models/optuna_classifier.py
"""

import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import fbeta_score
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

from models.resnet import SokobanSEResNetClassifier, ClassifierLoss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
N_TRIALS    = 30
FOLD        = 1
MAX_EPOCHS  = 20     # Reducido por trial (el pruner corta los malos antes)
PATIENCE    = 5
BETA        = 0.5    # F_0.5 premia la precisión (evitar falsos positivos)

db_url     = os.environ.get("OPTUNA_DB_URL",     f"sqlite:///{RESULTS_DIR}/optuna_classifier.db")
study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*55}")
print(f"  OPTUNA — Surrogate Classifier (Fold {FOLD})")
print(f"  Dispositivo: {device.type.upper()}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Trials: {N_TRIALS} | Max épocas/trial: {MAX_EPOCHS}")
print(f"  Métrica objetivo: F_{BETA} (precision > recall)")
print(f"  Storage: {db_url}")
print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class FoldDataset(Dataset):
    def __init__(self, data_dict):
        self.tensors = data_dict["tensor"]
        self.labels = data_dict["is_solvable"]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.tensors[idx].float(), self.labels[idx].float()


# Cargar datos UNA sola vez (no recargar en cada trial)
print(f"Cargando Fold {FOLD} (puede tardar ~30s)...")
_train_data = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_train.pt", weights_only=False)
_val_data   = torch.load(f"{RESULTS_DIR}/classifier_fold{FOLD}_val.pt",  weights_only=False)

# Calcular pesos para BCEWithLogitsLoss
try:
    _N_pos = (_train_data["is_solvable"] == 1).sum().item()
    _N_neg = len(_train_data["is_solvable"]) - _N_pos
    _pos_weight_val = _N_neg / max(1, _N_pos)
except Exception:
    _pos_weight_val = 1.0

print(f"Train: {len(_train_data['is_solvable']):,} | Validation: {len(_val_data['is_solvable']):,}")
print(f"Solubles: {_N_pos:,} | Deadlocks: {_N_neg:,} | pos_weight base: {_pos_weight_val:.2f}\n")

_val_dataset = FoldDataset(_val_data)


def make_loaders(batch_size):
    from torch.utils.data import RandomSampler
    ds_train = FoldDataset(_train_data)
    # Entrenar solo con 30k muestras por época para que Optuna sea rápido
    sampler = RandomSampler(ds_train, replacement=True, num_samples=min(30000, len(ds_train)))
    train_loader = DataLoader(ds_train, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    
    # Validar solo con 5k muestras
    subset_val_data = {
        "tensor": _val_data["tensor"][:5000],
        "is_solvable": _val_data["is_solvable"][:5000]
    }
    val_loader   = DataLoader(FoldDataset(subset_val_data), batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial):
    lr           = trial.suggest_float("lr",           3e-4, 8e-4,  log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3,  log=True)
    dropout_p    = trial.suggest_float("dropout_p",    0.25, 0.35)
    # pos_weight < 1 → penaliza menos los falsos positivos (más precisión)
    # pos_weight > 1 → penaliza menos los falsos negativos (más recall)
    pos_weight   = trial.suggest_float("pos_weight",   5.0,  8.0)
    batch_size   = trial.suggest_categorical("batch_size", [64, 128])

    print(f"\n[Trial {trial.number}] lr={lr:.5f} | wd={weight_decay:.6f} | "
          f"drop={dropout_p:.2f} | pw={pos_weight:.2f} | bs={batch_size}")

    train_loader, val_loader = make_loaders(batch_size)

    model     = SokobanSEResNetClassifier(dropout_p=dropout_p).to(device)
    criterion = ClassifierLoss(pos_weight_val=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f_beta  = 0.0
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        epoch_logits = []
        for tensors, labels in train_loader:
            tensors = tensors.to(device)
            labels  = labels.to(device)
            optimizer.zero_grad()
            logits  = model(tensors)
            epoch_logits.extend(logits.detach().cpu().numpy())
            loss    = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        # Logging temporal para ver si hay colapso de varianza
        if epoch == 1 or epoch % 5 == 0:
            el_arr = np.array(epoch_logits)
            print(f"  [Epoch {epoch}] logits: mean={el_arr.mean():.3f} std={el_arr.std():.3f}")

        # ── Eval ─────────────────────────────────────────────────────────────
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for tensors, labels in val_loader:
                tensors = tensors.to(device)
                logits  = model(tensors)
                probs   = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(labels.numpy())

        # Barrido de umbrales
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        best_epoch_f_beta = 0.0
        best_epoch_thresh = 0.5
        
        for thresh in np.arange(0.50, 0.96, 0.05):
            preds = (all_probs >= thresh).astype(float)
            fb = fbeta_score(all_targets, preds, beta=BETA, zero_division=0)
            if fb > best_epoch_f_beta:
                best_epoch_f_beta = fb
                best_epoch_thresh = thresh

        scheduler.step()

        print(f"  [Trial {trial.number}] Época {epoch:02d} | Max F_{BETA}={best_epoch_f_beta:.4f} (Umbral {best_epoch_thresh:.2f})")

        # Optuna Pruner: cancela trials malos temprano
        trial.report(best_epoch_f_beta, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if best_epoch_f_beta > best_f_beta:
            best_f_beta  = best_epoch_f_beta
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  [Trial {trial.number}] Early stopping en época {epoch}.")
                break

    return best_f_beta


# ─────────────────────────────────────────────────────────────────────────────
# ESTUDIO OPTUNA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="maximize",   # queremos F_0.5 lo más alto posible
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=study_name,
        storage=optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600}
        ),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*55)
    print("  RESULTADOS FINALES DEL ENJAMBRE — CLASIFICADOR")
    print("="*55)
    print(f"  Total de Trials: {len(study.trials)}")
    print(f"  Mejor F_{BETA}: {study.best_value:.4f}")
    print(f"  Mejores hiperparámetros:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    out_path = os.path.join(RESULTS_DIR, "best_hparams_classifier.json")
    with open(out_path, "w") as f:
        json.dump({"best_f_beta": study.best_value, "beta": BETA,
                   "params": study.best_params}, f, indent=2)
    print(f"\n  ✅ Hiperparámetros guardados en: {out_path}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n  Completados: {len(completed)} | Podados: {len(pruned)}")
