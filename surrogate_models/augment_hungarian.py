import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from tqdm import tqdm

# ─── Esquema de canales (confirmado contra board_utils.py y neural_heuristic.cpp) ───
# C0 = muros
# C1 = piso/interior caminable
# C2 = cajas
# C3 = metas         ← CORRECCIÓN vs versión anterior que usaba canal 1 por error
# C4 = jugador
# C5 = deadlock mask

CHANNEL_WALLS = 0
CHANNEL_GOALS = 3  # ← Fix crítico: era 1 (piso), debe ser 3 (metas)
CHANNEL_BOXES = 2


def get_hungarian_lb(tensor_board):
    """
    Calcula el Hungarian Lower Bound para un tablero (6, 25, 25).

    Para cada par (caja, meta) calcula la distancia Manhattan mínima con BFS
    ignorando muros, y resuelve la asignación óptima (mínimo costo total) con
    el algoritmo húngaro (linear_sum_assignment de scipy).

    Esquema de canales real:
        C0=muros, C1=piso, C2=cajas, C3=metas, C4=jugador, C5=deadlock_mask
    """
    walls = tensor_board[CHANNEL_WALLS].numpy() == 1
    goals = np.argwhere(tensor_board[CHANNEL_GOALS].numpy() == 1)  # C3
    boxes = np.argwhere(tensor_board[CHANNEL_BOXES].numpy() == 1)  # C2

    if len(goals) == 0 or len(boxes) == 0 or len(goals) != len(boxes):
        return 0.0

    H, W = walls.shape

    # BFS desde cada meta hacia todas las celdas alcanzables
    dist_maps = []
    for gr, gc in goals:
        dist = np.full((H, W), np.inf)
        dist[gr, gc] = 0
        q = deque([(gr, gc)])
        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not walls[nr, nc] and np.isinf(dist[nr, nc]):
                        dist[nr, nc] = d + 1
                        q.append((nr, nc))
        dist_maps.append(dist)

    # Construir matriz de costos Goal×Box
    n = len(goals)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j, (br, bc) in enumerate(boxes):
            cost_matrix[i, j] = dist_maps[i][br, bc]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    lb = cost_matrix[row_ind, col_ind].sum()

    # Si alguna caja es inalcanzable desde una meta → lb = inf → devolvemos 0
    if np.isinf(lb):
        return 0.0
    return float(lb)


def _purge_stale_keys(item: dict) -> dict:
    """Elimina claves generadas por la versión bugeada para forzar recálculo."""
    for key in ('hungarian_lb', 'residual_raw', 'residual_norm'):
        item.pop(key, None)
    return item


def process_file(filepath, force_recompute: bool = False):
    """
    Calcula hungarian_lb y residual_raw para cada tablero del archivo.

    force_recompute=True: borra las claves existentes y recalcula desde cero.
    Úsalo si el archivo ya fue procesado con la versión bugeada (canal 1 en
    vez de canal 3 para las metas).
    """
    print(f"Procesando {filepath}  (force_recompute={force_recompute})...")
    data = torch.load(filepath, weights_only=False)

    for i in tqdm(range(len(data)), desc="Hungarian LB"):
        if force_recompute:
            _purge_stale_keys(data[i])

        if 'hungarian_lb' not in data[i]:
            lb = get_hungarian_lb(data[i]['tensor'])
            data[i]['hungarian_lb'] = lb
            # residual ≥ 0 por construcción (Hungarian LB es una cota inferior)
            data[i]['residual_raw'] = max(0.0, data[i]['pushes_raw'] - lb)

    torch.save(data, filepath)
    print(f"  ✅ Guardado: {filepath}")


def main(force_recompute: bool = False):
    results_dir = "surrogate_models/results"

    # ── Paso 1: calcular lb + residual_raw en train/val/test de los 5 folds ──
    for fold in range(1, 6):
        for split in ('train', 'val', 'test'):
            f = os.path.join(results_dir, f"regressor_fold{fold}_{split}.pt")
            if os.path.exists(f):
                process_file(f, force_recompute=force_recompute)

    # ── Paso 2: calcular estadísticas del residual sobre TRAIN y normalizarlo ──
    for fold in range(1, 6):
        train_path = os.path.join(results_dir, f"regressor_fold{fold}_train.pt")
        stats_path = os.path.join(results_dir, f"regressor_fold{fold}_stats.pt")
        if not (os.path.exists(train_path) and os.path.exists(stats_path)):
            continue

        train_data = torch.load(train_path, weights_only=False)
        stats = torch.load(stats_path, weights_only=False)

        residuals_log = [np.log1p(d['residual_raw']) for d in train_data]
        r_mean = float(np.mean(residuals_log))
        r_std  = float(np.std(residuals_log))

        # Protección contra std=0 (improbable, pero seguro)
        if r_std < 1e-8:
            print(f"  ⚠️  Fold {fold}: std del residual ~0, revisar datos.")
            r_std = 1.0

        stats['residual_mean'] = r_mean
        stats['residual_std']  = r_std

        for i in range(len(train_data)):
            train_data[i]['residual_norm'] = (
                np.log1p(train_data[i]['residual_raw']) - r_mean
            ) / r_std

        torch.save(train_data, train_path)
        torch.save(stats, stats_path)

        # Aplicar la misma normalización a val y test (usando stats de train)
        for split in ('val', 'test'):
            split_path = os.path.join(results_dir, f"regressor_fold{fold}_{split}.pt")
            if not os.path.exists(split_path):
                continue
            split_data = torch.load(split_path, weights_only=False)
            for i in range(len(split_data)):
                split_data[i]['residual_norm'] = (
                    np.log1p(split_data[i]['residual_raw']) - r_mean
                ) / r_std
            torch.save(split_data, split_path)

        # Sanity-check: ¿el lb promedio es razonable?
        lb_vals = [d['hungarian_lb'] for d in train_data]
        res_vals = [d['residual_raw'] for d in train_data]
        print(
            f"  ✅ Fold {fold} | Hungarian LB: avg={np.mean(lb_vals):.1f}  min={np.min(lb_vals):.0f}  max={np.max(lb_vals):.0f}"
            f" | Residual: avg={np.mean(res_vals):.1f} | Residual log-std={r_std:.3f}"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Borra claves hungarian_lb/residual_raw/residual_norm existentes y "
             "recalcula desde cero. Necesario si el script ya corrió con el bug "
             "del canal equivocado (C1 en vez de C3 para las metas)."
    )
    args = parser.parse_args()
    main(force_recompute=args.force_recompute)
