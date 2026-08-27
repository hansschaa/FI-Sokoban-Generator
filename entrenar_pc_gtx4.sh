#!/bin/bash
set -e

# ==============================================================================
# SCRIPT DE REENTRENAMIENTO: PATH CONSISTENCY (100% DATOS) - PARA GTX_4
# ==============================================================================
echo "🚀 Iniciando proceso de reentrenamiento completo de Path Consistency..."

# 0. Activar entorno virtual
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "⚠️ Advertencia: No se encontró venv/bin/activate. Se asume entorno global."
fi

# 1. Calibración del Pruner
echo ""
echo "=== FASE 1: Calibración del MedianPruner ==="
python3 surrogate_models/run_pruner_calibration.py

# Extraer el warmup recomendado del JSON generado
CALIBRATION_FILE="surrogate_models/results/pruner_calibration_results.json"
WARMUP=$(python3 -c "import json; print(json.load(open('$CALIBRATION_FILE'))['recommended_warmup'])")
echo "✅ Warmup detectado automáticamente para Optuna: $WARMUP épocas"

# 2. Búsqueda de Hiperparámetros (Optuna) sobre el Fold 3
echo ""
echo "=== FASE 2: Búsqueda Optuna (Fold 3) ==="
python3 surrogate_models/optuna_path_consistency_v2.py \
    --study-name pc_optuna_fold3 \
    --n-trials 50 \
    --fold 3 \
    --pruner-warmup $WARMUP

# 3. Entrenamiento al 100% con los hiperparámetros ganadores
echo ""
echo "=== FASE 3: Amalgamiento Final (100% de los Folds) ==="
# El script v2 toma automáticamente los hiperparámetros de best_hparams_path_consistency_v2.json 
# (generado por optuna) si no se los pasamos manualmente.
python3 surrogate_models/train_final_path_consistency_v2.py \
    --folds 1,2,3,4,5 \
    --epochs 25 \
    --output surrogate_models/results/path_consistency/production_path_consistency.pt \
    --restart

# 4. Exportar a TorchScript (C++)
echo ""
echo "=== FASE 4: Exportación JIT para Motor C++ ==="
python3 scratch/export_production_pc_to_jit.py \
    --model-path surrogate_models/results/path_consistency/production_path_consistency.pt \
    --out-jit surrogate_models/results/production_path_consistency_jit.pt

echo ""
echo "🎉 ¡PROCESO COMPLETADO! El modelo final (optimizado con sus propios hparams) está listo en: surrogate_models/results/production_path_consistency_jit.pt"
