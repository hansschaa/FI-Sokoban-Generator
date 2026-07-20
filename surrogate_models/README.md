# Surrogate Models para Sokoban

Pipeline limpio para entrenar los modelos sustitutos del generador evolutivo.

## Estructura

```
surrogate_models/
├── data/                         # Scripts de preparación de datos
│   └── prepare_regressor.py      # Parseo, shell_hash, GroupKFold, tensores
├── models/                       # Definición de arquitecturas PyTorch
│   └── resnet.py                 # SimpleResNet adaptada (Regressor + Classifier heads)
├── notebooks/                    # Jupyter Notebooks de entrenamiento y análisis
│   └── train_regressor.ipynb     # Entrenamiento con 5-Fold + métricas
├── results/                      # Outputs: pesos .pt, gráficos, CSVs de métricas
└── README.md
```

## Datos de entrada
- **Solubles**: `../training_data/Solvables/<pc>/<bucket>/*.sok`
- **Insolubles**: `../training_data/Unsolvables/deadlocks.sok`

## Orden de ejecución
1. `python data/prepare_regressor.py`  → genera los 5 folds en `results/`
2. Ejecutar `notebooks/train_regressor.ipynb`
