# Resumen de Estado del Proyecto - Sokoban Generator (Agosto 2026)

Este documento centraliza toda la información crítica del proyecto para facilitar su retoma en el futuro. Todos los experimentos, métricas finales y modelos entrenados descritos en el paper se encuentran alojados en la computadora **GTX_4** del laboratorio.

## 1. Hardware del Laboratorio (GTX_4)
- **CPU:** Intel Core Ultra 9 285K (24 núcleos, 24 hilos, 3.7 GHz base)
- **RAM:** 64 GB
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Arquitectura Blackwell)
- **Rol:** Entrenamiento de modelos en PyTorch y ejecución del motor C++ (A* Batched y Evolutivo).

## 2. Modelos de Producción (Checkpoints `.pt`)

El sistema consta de 3 redes neuronales convolucionales (SE-ResNet) principales. 

> [!IMPORTANT]
> **Falta completar:** Añadir los nombres exactos de los archivos `.pt` y sus hashes (SHA-256) para garantizar trazabilidad.

### A. Regresor Original (Evolutivo)
- **Función:** Estimador de *fitness* (longitud de solución) en la metaheurística Evolutiva ($\mu + \lambda$).
- **Datos de Entrenamiento:** Corpus Combinado (Original + Denso).
- **Partición:** Modelo ganador del **Fold 4** (Validación Cruzada de 5 folds).
- **Optimización (Optuna):** Guiado por la minimización del **MAE** sobre el conjunto de validación.
- **Archivo en GTX_4:** `production_regressor.pt` (o su equivalente original `final_regressor_fold4.pt`)
- **SHA-256:** `26bef55169151fc2ea22696ffc6c4dbc06947cdab83cfe07f194fc559903e064`

### B. Clasificador Contrastivo (Auxiliar)
- **Función:** Penalizador de tableros injugables (*deadlocks*) dentro del motor evolutivo.
- **Datos de Entrenamiento:** Corpus Combinado (pares positivos/negativos).
- **Optimización (Optuna):** Guiado por el score **F0.5** sobre el subconjunto de validación denso.
- **Archivo en GTX_4:** `production_contrastive_classifier_v2_combined.pt`
- **SHA-256:** `01fd3053cf2c8474db7226e3530d2e5c96c983b2b87d4a0fc928d2a4f6b0b12b`

### C. Regresor Path Consistency (A* Batched Masivo)
- **Función:** Heurística informada para el motor C++ masivo (`BATCH_K=64`). Logró un speedup de 14x y redujo los nodos expandidos físicos a un promedio de 80.6.
- **Datos de Entrenamiento:** Corpus Combinado (Original + Denso).
- **Partición:** Modelo ganador del **Fold 3** (Validación Cruzada de 5 folds).
- **Optimización:** Reutiliza los hiperparámetros base del regresor original, agregando $\alpha$ y `margin` manualmente para la arquitectura siamesa.
- **Archivo en GTX_4:** `final_path_consistency_production_jit.pt` (Exportado para C++)
- **SHA-256:** `[EL HASH DEBES SACARLO DESDE LA GTX_4 DIRECTAMENTE]`

## 3. Hiperparámetros Base (Arquitectura SE-ResNet)
Los hiperparámetros fundamentales seleccionados por Optuna (MedianPruner) para los regresores fueron:
- `learning_rate`: [Valor]
- `batch_size`: 256
- `weight_decay`: [Valor]
- `num_blocks`: [Valor]
- `channels`: [Valor]

## 4. Conjuntos de Datos Clave
- **Corpus Combinado:** Fusión del corpus original (Sokoban estándar) y el corpus minado denso (topologías restrictivas). Usado en K-Fold de 5 particiones.
- **Benchmark Held-Out (N=40):** Tableros de prueba (`benchmark_stratified_heldout.sok`) estrictamente aislados y nunca vistos en entrenamiento. 
- **Intersección Estricta (N=17):** Subconjunto de tableros resueltos exitosamente por TODAS las heurísticas (incluyendo Manhattan) para asegurar evaluación justa (Tabla III del paper).

## 5. Trabajo Futuro Pendiente
Si se retoma el proyecto, las líneas inmediatas de acción (declaradas en el paper) son:
1. **Entrenamiento Final:** Entrenar los modelos de producción (Regresor Evolutivo y Path Consistency) sobre el **100% de la base de datos combinada** (sin dejar folds ocultos) para exprimir el techo arquitectónico.
2. **Exploración de Metaheurísticas:** Extender el uso de los modelos subrogados a Algoritmos Genéticos (con crossover) y *Simulated Annealing*.
3. **Pérdida Orientada a Rank:** Formalizar la función de pérdida del modelo contrastivo (o del regresor) con optimización directa de métricas de ordenamiento por pares (Pairwise Ranking).

---
*Documento generado al cierre de la redacción del paper (Agosto 2026).*
