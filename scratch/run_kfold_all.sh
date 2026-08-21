#!/bin/bash
set -e

echo "=== Iniciando K-Fold Cross Validation (Path Consistency V2) ==="

# Crear directorio para guardar los logs y los pesos de test
mkdir -p surrogate_models/results/path_consistency/kfold

for fold in 1 2 3 4 5; do
    echo "----------------------------------------------------"
    echo "🚀 Entrenando con Test Fold = $fold"
    echo "----------------------------------------------------"
    
    # Salida del modelo
    OUT_MODEL="surrogate_models/results/path_consistency/kfold/path_consistency_test_fold${fold}.pt"
    
    # Log file
    LOG_FILE="surrogate_models/results/path_consistency/kfold/train_fold${fold}.log"
    
    # Entrenar (usamos stdbuf para forzar flush al log en tiempo real)
    stdbuf -oL -eL python3 surrogate_models/train_kfold_path_consistency_v2.py \
        --test-fold $fold \
        --output $OUT_MODEL \
        | tee $LOG_FILE
        
    echo "✅ Fold $fold completado. Log guardado en $LOG_FILE"
done

echo "🎉 K-Fold Cross Validation Finalizado Exitosamente!"
