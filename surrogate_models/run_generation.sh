#!/bin/bash
set -e

echo "1. Actualizando repositorio..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"
git pull

cd surrogate_models
echo "2. Regenerando datasets para los 5 Folds en paralelo (10 workers)..."
for i in {0..9}; do
    ../venv/bin/python3 prepare_path_consistency.py --part $i --total-parts 10 > /dev/null 2>&1 &
done
wait
echo "   -> Generación terminada."

echo "3. Fusionando partes..."
cat << 'EOF' > merge_parts.py
import torch
import os

for fold in range(1, 6):
    all_pairs = []
    for part in range(10):
        path = f"results/path_consistency/path_fold{fold}_train_part{part}.pt"
        if os.path.exists(path):
            pairs = torch.load(path, weights_only=False, map_location='cpu')
            all_pairs.extend(pairs)
    
    out_path = f"results/path_consistency/path_fold{fold}_train.pt"
    if len(all_pairs) > 0:
        torch.save(all_pairs, out_path)
        print(f"Merged Fold {fold}: {len(all_pairs)} pairs saved to {out_path}")
EOF
../venv/bin/python3 merge_parts.py
rm merge_parts.py

cd ..
echo "4. Verificando dataset..."
./venv/bin/python3 surrogate_models/verify_dataset.py

echo "5. Ejecutando 1 solo trial de Optuna como verificación..."
cd surrogate_models
../venv/bin/python3 optuna_path_consistency.py --n-trials 1

echo "=========================================================="
echo "PAUSA DE SEGURIDAD."
echo "Verifica el output de arriba:"
echo " (a) pushes1 y pushes2 varían en verify_dataset."
echo " (b) Inter-branch Acc no es exactamente 0.0 ni 1.0."
echo "Si todo está OK, ejecuta: ./venv/bin/python3 surrogate_models/optuna_path_consistency.py"
echo "=========================================================="
