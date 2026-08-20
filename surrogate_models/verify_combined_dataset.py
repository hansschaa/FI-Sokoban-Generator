"""
PASO 1 — Verificación de balance del dataset combinado (Original + Denso).
Correr ANTES del entrenamiento para confirmar que los tensores son correctos.

Uso:
  cd ~/hans/FI-Sokoban-Generator
  source venv/bin/activate
  python3 surrogate_models/verify_combined_dataset.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def main():
    print("=" * 70)
    print("  VERIFICACIÓN DE BALANCE — Dataset Contrastivo Combinado (Fold 0)")
    print("=" * 70)

    files = {
        "X_train": "contrastive_fold_0_X_train.pt",
        "y_train": "contrastive_fold_0_y_train.pt",
        "t_train": "contrastive_fold_0_t_train.pt",
        "X_test":  "contrastive_fold_0_X_test.pt",
        "y_test":  "contrastive_fold_0_y_test.pt",
        "t_test":  "contrastive_fold_0_t_test.pt",
    }

    # 1. Verificar existencia y fechas
    print("\n📁 Archivos y fechas de modificación:")
    from datetime import datetime
    all_exist = True
    for label, fname in files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            size_mb = os.path.getsize(path) / 1e6
            print(f"  ✅ {fname}: {size_mb:,.1f} MB | {mtime}")
        else:
            print(f"  ❌ {fname}: NO EXISTE")
            all_exist = False

    if not all_exist:
        print("\n❌ Faltan archivos. Corré prepare_contrastive_classifier.py primero.")
        return

    # 2. Cargar tensores
    print("\n📊 Cargando tensores...")
    y_train = torch.load(os.path.join(RESULTS_DIR, files["y_train"]), map_location="cpu", weights_only=False)
    t_train = torch.load(os.path.join(RESULTS_DIR, files["t_train"]), map_location="cpu", weights_only=False)
    X_train = torch.load(os.path.join(RESULTS_DIR, files["X_train"]), map_location="cpu", weights_only=False)

    y_test = torch.load(os.path.join(RESULTS_DIR, files["y_test"]), map_location="cpu", weights_only=False)
    t_test = torch.load(os.path.join(RESULTS_DIR, files["t_test"]), map_location="cpu", weights_only=False)
    X_test = torch.load(os.path.join(RESULTS_DIR, files["X_test"]), map_location="cpu", weights_only=False)

    # 3. Reportar shapes
    print(f"\n  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    assert X_train.shape[1:] == (12, 25, 25), f"ERROR: X_train tiene shape {X_train.shape}, esperado (N, 12, 25, 25)"
    assert X_test.shape[1:]  == (12, 25, 25), f"ERROR: X_test tiene shape {X_test.shape}, esperado (N, 12, 25, 25)"

    # 4. Balance por label (y)
    def report_balance(y, t, name):
        total = len(y)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        ratio = neg / max(1, pos)

        t1 = int((t == 1).sum())
        t2 = int((t == 2).sum())
        t3 = int((t == 3).sum())

        print(f"\n  {name} ({total:,} muestras):")
        print(f"    Label:  Positivos (Solubles)={pos:,} | Negativos (Deadlocks)={neg:,} | Ratio neg/pos={ratio:.3f}")
        print(f"    Tipo:   t==1 (Solvable)={t1:,} | t==2 (Simple DL)={t2:,} | t==3 (Complex DL)={t3:,}")

        # Cruce label x tipo
        for tv in [1, 2, 3]:
            mask = t == tv
            if mask.sum() > 0:
                y_sub = y[mask]
                p = int((y_sub == 1).sum())
                n = int((y_sub == 0).sum())
                print(f"      t=={tv}: pos={p:,}, neg={n:,}")

    report_balance(y_train, t_train, "TRAIN")
    report_balance(y_test, t_test, "TEST")

    # 5. Combinado (lo que verá el modelo de producción)
    y_all = torch.cat([y_train, y_test])
    t_all = torch.cat([t_train, t_test])
    report_balance(y_all, t_all, "COMBINADO (Train + Test = 100%)")

    total_all = len(y_all)
    print(f"\n  ✅ TOTAL MUESTRAS PARA ENTRENAMIENTO: {total_all:,}")
    print(f"  ✅ Canales de entrada: {X_train.shape[1]} (esperado: 12)")
    print(f"  ✅ Tamaño espacial: {X_train.shape[2]}x{X_train.shape[3]} (esperado: 25x25)")

    # 6. Sanity check: ¿hay tipos densos?
    has_dense = int((t_all == 2).sum()) > 0 or int((t_all == 3).sum()) > 0
    if has_dense:
        print(f"\n  ✅ CONFIRMADO: Dataset incluye deadlocks densos (Simple + Complex).")
    else:
        print(f"\n  ⚠️  ATENCIÓN: NO hay deadlocks densos (t==2/t==3). ¿Se regeneraron los tensores con datos densos?")

    print("\n" + "=" * 70)
    print("  Si todo se ve correcto, procedé al PASO 2: Entrenamiento.")
    print("=" * 70)

if __name__ == "__main__":
    main()
