#!/bin/bash
set -e

echo "=== Preparando modelo V2 para Ablation Study ==="
MODEL_PATH="surrogate_models/results/path_consistency/final_path_consistency_production.pt"
JIT_PATH="surrogate_models/results/surrogate_regressor_jit.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: No se encontró el modelo entrenado en $MODEL_PATH"
    echo "Asegúrate de que train_final_path_consistency_v2.py haya terminado."
    exit 1
fi

echo "1. Exportando $MODEL_PATH a JIT..."
source venv/bin/activate
python scratch/export_jit.py --model-path $MODEL_PATH --output-path $JIT_PATH

echo "2. Reiniciando Surrogate Server (si estaba corriendo)..."
pkill -f surrogate_server.py || true

echo "✅ Todo listo. Puedes lanzar el Ablation Study con:"
echo "   python scratch/run_exp1_2x2_matrix.py"
