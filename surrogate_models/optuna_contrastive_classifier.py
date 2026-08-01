"""
optuna_contrastive_classifier.py
--------------------------------
Búsqueda distribuida de hiperparámetros con Optuna para el Clasificador Contrastivo (12 canales).
Enfocado en perfeccionar la precisión en el régimen < 6 cajas para reducir disparos en mapas extensos (shell_1).

- Usa el Fold 1 (contrastive_fold_0_*.pt) como conjunto de búsqueda/validación.
- Hiperparámetros de entrenamiento explorados (4 dimensiones):
    * lr (1e-5 a 5e-3, log-scale)
    * weight_decay (1e-6 a 1e-2, log-scale)
    * dropout_p (0.10 a 0.50)
    * batch_size ([64, 128, 256, 512])
- Evaluación Óptima Sin Reentrenar: El umbral de decisión se determina dinámicamente mediante barrido 
  posterior en cada época ([0.50, 0.95]), permitiendo que cada trial reporte su verdadero óptimo de calibración.
- Métrica objetivo: Máximo F_0.5 en el barrido de umbral (priorizando precisión y evitando falsos positivos).
- Poda inteligente y muy permisiva: MedianPruner con n_warmup_steps=9 (de 15 épocas) para evitar podar durante
  la fase inicial de convergencia y dar tiempo a que los modelos converjan y revelen su potencial real.

Ejecutar en el clúster de laboratorio:
    export OPTUNA_DB_URL="mysql+pymysql://USER:PASSWORD@HOST/optuna_db"
    export OPTUNA_STUDY_NAME="sokoban_contrastive_lab_v3"
    venv/bin/python surrogate_models/optuna_contrastive_classifier.py
"""

import sys, os, json, gc, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.pruners import MedianPruner, NopPruner
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

from models.resnet import SokobanSEResNetClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
N_TRIALS    = 50
FOLD        = 1
MAX_EPOCHS  = 15
PATIENCE    = 4
BETA        = 0.5    # F_0.5 prioriza precisión para minimizar falsos positivos del disyuntor
WARMUP_EPOCHS = 11   # 11 épocas de gracia antes de podar para superar caídas transitorias de ruido hasta la Ep 10

db_url     = os.environ.get("OPTUNA_DB_URL", f"sqlite:///{RESULTS_DIR}/optuna_contrastive_classifier.db")
study_name = os.environ.get("OPTUNA_STUDY_NAME", "sokoban_contrastive_lab_v3")
no_prune   = os.environ.get("OPTUNA_NO_PRUNED", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*65}")
print(f"  OPTUNA — Clasificador Contrastivo (12 canales, Fold {FOLD})")
print(f"  Dispositivo: {device.type.upper()}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Trials: {N_TRIALS} | Max épocas/trial: {MAX_EPOCHS} | Warmup Pruning: {'DESACTIVADO (Control)' if no_prune else f'{WARMUP_EPOCHS} épocas'}")
print(f"  Métrica objetivo: F_{BETA} óptimo post-barrido (Precisión > Recall)")
print(f"  Storage: {db_url}")
print(f"{'='*65}\n")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONTRASTIVO CON D4 AUGMENTATION EN TRAIN
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveMemoryDataset(Dataset):
    def __init__(self, X_tensor, y_tensor, t_tensor, is_train=False):
        self.X = X_tensor
        self.y = y_tensor
        self.t = t_tensor
        self.is_train = is_train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            # Rotaciones y flipes simétricos para los 12 canales simultáneamente
            k = random.randint(0, 3)
            flip = random.choice([True, False])
            x = torch.rot90(x, k, [1, 2])
            if flip:
                x = torch.flip(x, [2])
        return x, self.y[idx], self.t[idx]

# Cargar en memoria una sola vez para máxima velocidad y eficiencia del clúster
print(f"Cargando dataset contrastivo Fold {FOLD} en memoria...")
train_X_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_X_train.pt")
train_y_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_y_train.pt")
train_t_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_t_train.pt")

val_X_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_X_test.pt")
val_y_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_y_test.pt")
val_t_path = os.path.join(RESULTS_DIR, f"contrastive_fold_{FOLD-1}_t_test.pt")

if not os.path.exists(train_X_path):
    raise FileNotFoundError(f"No se encontraron archivos del Fold {FOLD-1} en {RESULTS_DIR}. ¡Verificar rutas!")

_train_X = torch.load(train_X_path, map_location='cpu')
_train_y = torch.load(train_y_path, map_location='cpu')
_train_t = torch.load(train_t_path, map_location='cpu')

_val_X = torch.load(val_X_path, map_location='cpu')
_val_y = torch.load(val_y_path, map_location='cpu')
_val_t = torch.load(val_t_path, map_location='cpu')

num_pos = (_train_y == 1).sum().item()
num_neg = (_train_y == 0).sum().item()
_pos_weight_val = num_neg / max(1, num_pos)

print(f"Train: {len(_train_y):,} | Validación: {len(_val_y):,}")
print(f"Solubles: {num_pos:,} | Deadlocks: {num_neg:,} | pos_weight calculado: {_pos_weight_val:.3f}\n")

_train_dataset = ContrastiveMemoryDataset(_train_X, _train_y, _train_t, is_train=True)
_val_dataset   = ContrastiveMemoryDataset(_val_X, _val_y, _val_t, is_train=False)

def make_loaders(batch_size):
    train_loader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(_val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN OBJETIVO OPTUNA (4 Dimensiones)
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial):
    lr           = trial.suggest_float("lr",           1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_p    = trial.suggest_float("dropout_p",    0.10, 0.50)
    batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    print(f"\n[Trial {trial.number}] lr={lr:.5f} | wd={weight_decay:.6f} | drop={dropout_p:.2f} | bs={batch_size}")

    train_loader, val_loader = make_loaders(batch_size)

    model     = SokobanSEResNetClassifier(dropout_p=dropout_p, in_channels=12).to(device)
    pos_weight_tensor = torch.tensor([_pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_f_beta  = 0.0
    patience_ctr = 0
    best_stats   = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                logits  = model(X_batch)
                probs   = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(y_batch.numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        # Barrido de umbral posterior para encontrar el óptimo de la época sin reentrenar
        best_epoch_f05 = 0.0
        best_epoch_thresh = 0.5
        best_epoch_prec = 0.0
        best_epoch_rec = 0.0

        for thresh in np.arange(0.50, 0.96, 0.05):
            preds = (all_probs >= thresh).astype(float)
            fb = fbeta_score(all_targets, preds, beta=BETA, zero_division=0)
            if fb > best_epoch_f05:
                best_epoch_f05 = fb
                best_epoch_thresh = thresh
                best_epoch_prec = precision_score(all_targets, preds, zero_division=0)
                best_epoch_rec  = recall_score(all_targets, preds, zero_division=0)

        print(f"  [Trial {trial.number}] Época {epoch:02d}/{MAX_EPOCHS} | Max F0.5={best_epoch_f05:.4f} @ umbral={best_epoch_thresh:.2f} (Prec={best_epoch_prec:.3f}, Rec={best_epoch_rec:.3f})")

        # Optuna Pruner (podar solo tras la fase de warmup, si el F0.5 está por debajo de la mediana)
        trial.report(best_epoch_f05, epoch)
        if not no_prune and trial.should_prune():
            print(f"  [Trial {trial.number}] Podado en época {epoch} por MedianPruner (rendimiento inferior a mediana).")
            raise optuna.exceptions.TrialPruned()

        if best_epoch_f05 > best_f_beta:
            best_f_beta  = best_epoch_f05
            best_stats   = {
                "epoch": epoch, 
                "f05": best_epoch_f05, 
                "optimal_threshold": best_epoch_thresh,
                "precision": best_epoch_prec, 
                "recall": best_epoch_rec
            }
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  [Trial {trial.number}] Early stopping activado tras {patience_ctr} épocas sin mejora (Época {epoch}).")
                break

    trial.set_user_attr("best_stats", best_stats)
    return best_f_beta

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN OPTUNA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    pruner_instance = NopPruner() if no_prune else MedianPruner(n_startup_trials=10, n_warmup_steps=WARMUP_EPOCHS)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner_instance,
        study_name=study_name,
        storage=optuna.storages.RDBStorage(
            url=db_url,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 3600}
        ) if db_url.startswith("mysql") or db_url.startswith("postgresql") else db_url,
        load_if_exists=True,
    )

    print("Iniciando optimización del enjambre...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "="*65)
    print("  RESULTADOS FINALES DEL ENJAMBRE CONTRASTIVO")
    print("="*65)
    print(f"  Total de Trials en la DB: {len(study.trials)}")
    print(f"  Mejor F_{BETA}: {study.best_value:.4f}")
    print(f"  Mejores Hiperparámetros (Entrenamiento):")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    
    stats = study.best_trial.user_attrs.get("best_stats", {})
    print(f"  Calibración Óptima (Umbral inferido en barrido): {stats.get('optimal_threshold', 0.5):.2f}")
    print(f"  Métricas asociadas -> Precisión: {stats.get('precision', 0):.4f} | Recall: {stats.get('recall', 0):.4f}")

    out_path = os.path.join(RESULTS_DIR, "best_hparams_contrastive_classifier.json")
    with open(out_path, "w") as f:
        json.dump({
            "best_f_05": study.best_value,
            "best_params": study.best_params,
            "optimal_threshold": stats.get("optimal_threshold", 0.5),
            "best_trial_stats": stats
        }, f, indent=2)
    print(f"\n  ✅ Hiperparámetros y umbral óptimos exportados en: {out_path}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Estadísticas — Completados: {len(completed)} | Podados: {len(pruned)}")
