"""
optuna_path_consistency_v2.py
──────────────────────────────
Búsqueda de hiperparámetros Optuna para el regresor de Path Consistency.

Novedades respecto a la versión anterior:
  - α (peso de MarginRankingLoss) ahora está en el espacio de búsqueda.
  - MedianPruner con warmup configurable (--pruner-warmup) — NO asumir
    ningún valor; medirlo primero con run_pruner_calibration.py.
  - Logging de hash del pipeline al arranque para reproducibilidad.
  - Solo Fold 1 en esta primera pasada (--fold=1 por defecto).
  - --no-pruning: deshabilita toda poda (usado por run_pruner_calibration.py).

Uso típico:
  # Paso 0 (una sola vez): calibrar el warmup del pruner
  python surrogate_models/run_pruner_calibration.py

  # Paso 1: enjambre completo con el warmup medido
  python surrogate_models/optuna_path_consistency_v2.py \\
      --study-name pc_optuna_v2 --n-trials 50 --pruner-warmup <warmup_medido>
"""

import os, sys, hashlib, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.resnet import SokobanSEResNetRegressor
from train_final_path_consistency import PathConsistencyDataset, RESULTS_DIR
from evaluate_inter_branch import get_valid_children
from prepare_path_consistency import encode_board, simulate_path

import pandas as pd

# ── Logging de hash del pipeline ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

PIPELINE_FILES = [
    os.path.join(SCRIPT_DIR,   "models", "resnet.py"),
    os.path.join(SCRIPT_DIR,   "train_final_path_consistency.py"),
    os.path.join(SCRIPT_DIR,   "prepare_path_consistency.py"),
]

def compute_pipeline_hash():
    h = hashlib.sha256()
    for p in PIPELINE_FILES:
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(p.encode())
                h.update(f.read())
    return h.hexdigest()[:16]

PIPELINE_HASH = compute_pipeline_hash()
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_inter_branch(model, device, n_pairs: int = 500) -> float:
    """Evaluación inter-branch: ¿predice correctamente el orden óptimo vs subóptimo?"""
    model.eval()
    TSV_FILE  = os.path.join(SCRIPT_DIR, "results", "path_consistency_heldout.tsv")
    SOK_FILE  = os.path.join(PROJECT_ROOT, "sok_files", "benchmark_stratified_heldout.sok")
    BATCH_SOL = os.path.join(PROJECT_ROOT, "build", "batch_solver")

    if os.path.exists(TSV_FILE):
        try:
            df_check = pd.read_csv(TSV_FILE, sep='\t')
            if len(df_check) < 40:
                os.remove(TSV_FILE)
        except Exception:
            if os.path.exists(TSV_FILE): os.remove(TSV_FILE)

    if not os.path.exists(TSV_FILE):
        import subprocess
        cmd = [BATCH_SOL, SOK_FILE, "hungarian", TSV_FILE]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"[eval_inter_branch] WARN: no se pudo generar TSV: {e}")
            return 0.0

    df = pd.read_csv(TSV_FILE, sep='\t')
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    board_map = {}
    if os.path.exists(SOK_FILE):
        with open(SOK_FILE) as f: lines = f.readlines()
        name, board = None, []
        for line in lines:
            line = line.rstrip()
            if not line: continue
            if all(c in '# @$.*+-' for c in line):
                board.append(line)
            else:
                if name and board: board_map[name] = '\n'.join(board)
                name, board = line, []
        if name and board: board_map[name] = '\n'.join(board)

    total, correct = 0, 0
    with torch.no_grad():
        for _, row in df.iterrows():
            if total >= n_pairs: break
            if row['Status'] != 'SOLVED' or row['LURD_Path'] == 'NONE': continue
            name = str(row['LevelName']).split(' - ')[0].strip()
            if name not in board_map: continue

            states = simulate_path(board_map[name], row['LURD_Path'])
            for i in range(len(states) - 1):
                if total >= n_pairs: break
                s_curr, _ = states[i]
                s_opt,  _ = states[i+1]
                children  = get_valid_children(s_curr)
                siblings  = [c for c in children if c != s_opt]
                if not siblings: continue

                t_opt = torch.tensor(encode_board(s_opt)).unsqueeze(0).to(device)
                p_opt = model(t_opt).item()
                for s_sub in siblings:
                    t_sub = torch.tensor(encode_board(s_sub)).unsqueeze(0).to(device)
                    p_sub = model(t_sub).item()
                    total += 1
                    if p_opt < p_sub:
                        correct += 1

    return correct / total if total > 0 else 0.0


def make_objective(fold: int, n_eval_pairs: int, use_pruning: bool):
    def objective(trial: optuna.Trial) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Espacio de búsqueda (con alpha explícito) ────────────────────────
        lr           = trial.suggest_float("lr",           1e-5,  1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6,  1e-3, log=True)
        dropout_p    = trial.suggest_float("dropout_p",    0.0,   0.4)
        batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        alpha        = trial.suggest_float("alpha",        0.01,  0.5,  log=True)
        margin       = trial.suggest_float("margin",       0.01,  0.3)
        # ────────────────────────────────────────────────────────────────────

        model   = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
        dataset = PathConsistencyDataset(fold, augment=True, max_route_distance=1)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, drop_last=True)

        optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        huber_fn   = nn.HuberLoss(delta=1.0)
        ranking_fn = nn.MarginRankingLoss(margin=margin)

        stats  = torch.load(os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt"),
                            weights_only=False)
        p_mean = stats["pushes_mean"]
        p_std  = stats["pushes_std"]

        N_EPOCHS_OPTUNA = 10

        model.train()
        for epoch in range(N_EPOCHS_OPTUNA):
            t0 = time.time()
            epoch_loss = 0.0
            for batch in loader:
                x1     = batch['tensor1'].to(device)
                x2     = batch['tensor2'].to(device)
                p1_raw = batch['pushes1'].float().to(device)

                y_target = (torch.log1p(p1_raw) - p_mean) / p_std

                optimizer.zero_grad()
                combined = torch.cat([x1, x2], dim=0)
                pred     = model(combined).squeeze(-1)
                p_opt, p_sub = pred.split(x1.size(0))

                loss_huber  = huber_fn(p_opt, y_target)
                target_rank = torch.ones_like(p_opt)
                loss_rank   = ranking_fn(p_opt, p_sub, target_rank)
                loss        = loss_huber + alpha * loss_rank

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            elapsed = time.time() - t0
            print(f"  [Trial {trial.number}] Epoch {epoch+1}/{N_EPOCHS_OPTUNA} "
                  f"| Loss: {epoch_loss:.4f} | {elapsed:.1f}s")

            if use_pruning and epoch >= 3 and epoch % 2 == 1:
                acc = evaluate_inter_branch(model, device, n_pairs=200)
                trial.report(acc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        acc = evaluate_inter_branch(model, device, n_pairs=n_eval_pairs)
        print(f"  [Trial {trial.number}] Inter-branch Acc FINAL: {acc:.4f}")
        return acc

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Optuna search para Path Consistency regressor (v2)")
    parser.add_argument("--study-name",    type=str, default="pc_optuna_v2")
    parser.add_argument("--n-trials",      type=int, default=50)
    parser.add_argument("--fold",          type=int, default=1)
    parser.add_argument("--n-eval-pairs",  type=int, default=500)
    parser.add_argument("--pruner-warmup", type=int, default=5,
                        help="Warmup del MedianPruner. Medir con run_pruner_calibration.py")
    parser.add_argument("--no-pruning",    action="store_true",
                        help="Desactiva toda poda (usar para el piloto de calibración)")
    args = parser.parse_args()

    print("=" * 70)
    print("  OPTUNA PATH CONSISTENCY REGRESSOR v2")
    print(f"  Pipeline Hash : {PIPELINE_HASH}")
    print(f"  Fold          : {args.fold}")
    print(f"  N Trials      : {args.n_trials}")
    if args.no_pruning:
        print(f"  Pruning       : DESACTIVADO (modo piloto de calibración)")
    else:
        print(f"  Pruning       : MedianPruner(warmup={args.pruner_warmup})")
    print(f"  Eval Pairs    : {args.n_eval_pairs}")
    print("=" * 70)

    db_url = os.environ.get("OPTUNA_DB_URL", "sqlite:///path_consistency_v2.db")
    print(f"  DB            : {db_url}\n")

    pruner = (optuna.pruners.NopPruner() if args.no_pruning
              else optuna.pruners.MedianPruner(
                  n_startup_trials=5,
                  n_warmup_steps=args.pruner_warmup,
                  interval_steps=2))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=db_url,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner,
    )

    study.optimize(
        make_objective(fold=args.fold,
                       n_eval_pairs=args.n_eval_pairs,
                       use_pruning=not args.no_pruning),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 70)
    print("  RESULTADO FINAL")
    best = study.best_trial
    print(f"  Mejor Inter-branch Acc: {best.value:.4f}")
    print("  Hiperparámetros:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    out = {"trial": best.number, "value": best.value, "params": best.params}
    out_path = os.path.join(SCRIPT_DIR, "results", "best_hparams_path_consistency_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Hiperparámetros guardados en: {out_path}")


if __name__ == "__main__":
    main()
