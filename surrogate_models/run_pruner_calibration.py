"""
run_pruner_calibration.py
─────────────────────────
Piloto de calibración: corre N trials COMPLETOS (sin poda) para medir la
trayectoria real de convergencia del regresor de Path Consistency.

El objetivo es determinar en qué época el modelo empieza a separarse
estadísticamente entre trials buenos y malos, para configurar el warmup
del MedianPruner en el enjambre principal.

Salida:
  - Tabla de convergencia por trial impresa en consola.
  - Archivo: results/pruner_calibration_results.json con las trayectorias.
  - Recomendación de warmup basada en la varianza entre trials.

Uso:
  python surrogate_models/run_pruner_calibration.py [--n-trials 5] [--fold 1]
"""

import os, sys, json, hashlib, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.resnet import SokobanSEResNetRegressor
from train_final_path_consistency import PathConsistencyDataset, RESULTS_DIR
from evaluate_inter_branch import get_valid_children
from prepare_path_consistency import encode_board, simulate_path

import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Pipeline hash ──────────────────────────────────────────────────────────────
def compute_pipeline_hash():
    h = hashlib.sha256()
    for p in [os.path.join(SCRIPT_DIR, "models", "resnet.py"),
              os.path.join(SCRIPT_DIR, "train_final_path_consistency.py"),
              os.path.join(SCRIPT_DIR, "prepare_path_consistency.py")]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(p.encode()); h.update(f.read())
    return h.hexdigest()[:16]
PIPELINE_HASH = compute_pipeline_hash()
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_inter_branch(model, device, n_pairs=300) -> float:
    model.eval()
    TSV_FILE  = os.path.join(SCRIPT_DIR, "results", "path_consistency_heldout.tsv")
    SOK_FILE  = os.path.join(PROJECT_ROOT, "sok_files", "benchmark_stratified_heldout.sok")
    BATCH_SOL = os.path.join(PROJECT_ROOT, "build", "batch_solver")

    if os.path.exists(TSV_FILE):
        try:
            df_check = pd.read_csv(TSV_FILE, sep='\t')
            if len(df_check) < 40: os.remove(TSV_FILE)
        except: os.remove(TSV_FILE)

    if not os.path.exists(TSV_FILE):
        import subprocess
        try:
            subprocess.run([BATCH_SOL, SOK_FILE, "hungarian", TSV_FILE],
                           check=True, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"[eval] WARN: no se pudo generar TSV: {e}")
            return 0.0

    df = pd.read_csv(TSV_FILE, sep='\t').sample(frac=1.0, random_state=42).reset_index(drop=True)

    board_map = {}
    if os.path.exists(SOK_FILE):
        with open(SOK_FILE) as f: lines = f.readlines()
        name, board = None, []
        for line in lines:
            line = line.rstrip()
            if not line: continue
            if all(c in '# @$.*+-' for c in line): board.append(line)
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
                s_curr, _ = states[i]; s_opt, _ = states[i+1]
                siblings = [c for c in get_valid_children(s_curr) if c != s_opt]
                if not siblings: continue
                t_opt = torch.tensor(encode_board(s_opt)).unsqueeze(0).to(device)
                p_opt = model(t_opt).item()
                for s_sub in siblings:
                    t_sub = torch.tensor(encode_board(s_sub)).unsqueeze(0).to(device)
                    total += 1
                    if p_opt < model(t_sub).item(): correct += 1
    return correct / total if total > 0 else 0.0


def run_single_trial(trial_id: int, fold: int, rng_seed: int, device):
    """Entrena un trial completo (10 épocas) con hparams aleatorios y retorna
    la trayectoria de accuracy inter-branch por época."""
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    # Sampleo manual de hparams (misma distribución que Optuna)
    lr           = float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3))))
    weight_decay = float(np.exp(np.random.uniform(np.log(1e-6), np.log(1e-3))))
    dropout_p    = float(np.random.uniform(0.0, 0.4))
    batch_size   = int(np.random.choice([64, 128, 256, 512]))
    alpha        = float(np.exp(np.random.uniform(np.log(0.01), np.log(0.5))))
    margin       = float(np.random.uniform(0.01, 0.3))

    print(f"\n  [Trial {trial_id}] lr={lr:.2e} wd={weight_decay:.2e} "
          f"drop={dropout_p:.2f} bs={batch_size} alpha={alpha:.3f} margin={margin:.3f}")

    model   = SokobanSEResNetRegressor(dropout_p=dropout_p).to(device)
    dataset = PathConsistencyDataset(fold, augment=True, max_route_distance=1)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=True, drop_last=True)

    stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(RESULTS_DIR, "production_regressor_stats.pt")
        print(f"⚠️  {f'regressor_fold{fold}_stats.pt'} no encontrado — usando production_regressor_stats.pt")
    stats  = torch.load(stats_path, weights_only=False)
    p_mean = stats["pushes_mean"]
    p_std  = stats["pushes_std"]

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber_fn   = nn.HuberLoss(delta=1.0)
    ranking_fn = nn.MarginRankingLoss(margin=margin)

    N_EPOCHS = 10
    trajectory = []  # acc por época

    model.train()
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        epoch_loss = 0.0
        for batch in loader:
            x1 = batch['tensor1'].to(device)
            x2 = batch['tensor2'].to(device)
            p1_raw = batch['pushes1'].float().to(device)
            y_target = (torch.log1p(p1_raw) - p_mean) / p_std

            optimizer.zero_grad()
            combined = torch.cat([x1, x2], dim=0)
            pred = model(combined).squeeze(-1)
            p_opt, p_sub = pred.split(x1.size(0))

            loss = huber_fn(p_opt, y_target) + alpha * ranking_fn(
                p_opt, p_sub, torch.ones_like(p_opt))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        acc = evaluate_inter_branch(model, device, n_pairs=300)
        trajectory.append(acc)
        elapsed = time.time() - t0
        print(f"    Epoch {epoch+1:02d}/{N_EPOCHS} | Loss: {epoch_loss:.4f} "
              f"| Inter-branch Acc: {acc:.4f} | {elapsed:.1f}s")

    return {
        "trial_id": trial_id,
        "seed": rng_seed,
        "hparams": {
            "lr": lr, "weight_decay": weight_decay,
            "dropout_p": dropout_p, "batch_size": batch_size,
            "alpha": alpha, "margin": margin,
        },
        "trajectory": trajectory,
        "final_acc": trajectory[-1],
    }


def analyze_trajectories(results: list) -> int:
    """Analiza las trayectorias y recomienda el warmup del pruner."""
    print("\n" + "=" * 70)
    print("  ANÁLISIS DE CONVERGENCIA")
    print("=" * 70)

    n_epochs = len(results[0]["trajectory"])
    accs_by_epoch = [
        [r["trajectory"][ep] for r in results] for ep in range(n_epochs)
    ]

    print(f"\n  {'Época':>5}  {'Media':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  {'Separación?':>12}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")

    recommended_warmup = n_epochs  # conservador por defecto
    for ep in range(n_epochs):
        accs = accs_by_epoch[ep]
        mu, sigma, mn, mx = np.mean(accs), np.std(accs), np.min(accs), np.max(accs)
        sep = mx - mn
        # Consideramos que hay separación útil cuando std > 0.02 y rango > 0.05
        separado = "✅ Sí" if sigma > 0.02 and sep > 0.05 else "❌ No"
        if sigma > 0.02 and sep > 0.05 and recommended_warmup == n_epochs:
            recommended_warmup = ep  # primera época con separación
        print(f"  {ep+1:>5}  {mu:>8.4f}  {sigma:>8.4f}  {mn:>8.4f}  {mx:>8.4f}  {separado:>12}")

    print(f"\n  🎯 Warmup recomendado para MedianPruner: {recommended_warmup} épocas")
    print(f"     (Hasta esta época los trials no se diferencian estadísticamente)")
    return recommended_warmup


def main():
    parser = argparse.ArgumentParser(
        description="Piloto de calibración del pruner para Path Consistency")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Número de trials completos a correr (3-5 recomendado)")
    parser.add_argument("--fold",     type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  PILOTO DE CALIBRACIÓN — PATH CONSISTENCY PRUNER")
    print(f"  Pipeline Hash : {PIPELINE_HASH}")
    print(f"  Device        : {device.type.upper()}")
    print(f"  N Trials      : {args.n_trials} (sin poda, completos)")
    print(f"  Fold          : {args.fold}")
    print("=" * 70)
    print("\n  ATENCIÓN: estos trials NO se podan. El objetivo es medir la")
    print("  trayectoria real de convergencia antes de configurar MedianPruner.\n")

    seeds = [42, 137, 271, 404, 512][:args.n_trials]
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'─'*60}")
        print(f"  TRIAL {i+1}/{args.n_trials}  (seed={seed})")
        print(f"{'─'*60}")
        r = run_single_trial(i + 1, args.fold, seed, device)
        results.append(r)

    recommended_warmup = analyze_trajectories(results)

    # Guardar resultados
    out_path = os.path.join(SCRIPT_DIR, "results", "pruner_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "pipeline_hash": PIPELINE_HASH,
            "recommended_warmup": recommended_warmup,
            "trials": results,
        }, f, indent=2)

    print(f"\n  📁 Resultados guardados en: {out_path}")
    print(f"\n  ✅ Próximo paso:")
    print(f"     python surrogate_models/optuna_path_consistency_v2.py \\")
    print(f"         --study-name pc_optuna_v2 --n-trials 50 --pruner-warmup {recommended_warmup}")


if __name__ == "__main__":
    main()
