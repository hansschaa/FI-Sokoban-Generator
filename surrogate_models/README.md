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

## Entorno y Reproducibilidad (GPU Blackwell)
Para el entrenamiento en hardware de última generación (ej. RTX 5070 Ti, arquitectura Blackwell `sm_120`), la versión estable de PyTorch no cuenta con los binarios precompilados. Se utilizó específicamente la siguiente versión Nightly para asegurar compatibilidad y reproducibilidad de los resultados:

- **PyTorch:** `torch-2.12.0.dev20260408+cu128`
- **Instalación:** `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` (Excluyendo explícitamente `torchvision` para evitar conflictos en el resolver).
