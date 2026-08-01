#!/bin/bash
set -e

echo "=== 1. Entrenando Piloto Fold 1 ==="
./venv/bin/python3 surrogate_models/train_final_path_consistency.py --folds 1 --epochs 15 --alpha 0.1 --margin 0.05

echo "=== 2. Calculando Spearman Intra-Shell ==="
cat << 'PYEOF' > scratch/temp_spearman.py
import torch
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor
from scratch.run_pilot import diagnose_spearman

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SokobanSEResNetRegressor(dropout_p=0.02).to(device)
model.load_state_dict(torch.load('surrogate_models/results/path_consistency/final_regressor_fold1.pt', map_location=device, weights_only=False))

rho_boxes, rho_player = diagnose_spearman(model, device)
print(f"  Spearman (Fixed Player, Boxes Vary): {rho_boxes:.3f}")
print(f"  Spearman (Fixed Boxes, Player Vary): {rho_player:.3f}")
PYEOF
./venv/bin/python3 scratch/temp_spearman.py

echo "=== 3. Exportando a TorchScript ==="
./venv/bin/python3 surrogate_models/export_pc_to_jit.py --fold 1

echo "=== 4. Ejecutando Benchmark Estricto ==="
./venv/bin/python3 run_benchmark_stratified.py --meta scratch/meta_20_60_filtered.csv --sok scratch/sok_20_60.sok --out scratch/res_20_60_consistent.csv

echo "=== 5. Resumiendo Resultados del Benchmark ==="
./venv/bin/python3 scratch/intersection_benchmark.py scratch/res_20_60_consistent.csv

