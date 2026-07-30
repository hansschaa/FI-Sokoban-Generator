# Resultados Finales del Regresor: GPU Batching y Análisis del Colapso Intra-Bucket

Este documento consolida la evidencia final de la evaluación del modelo subrogado (Path Consistency) implementado vía GPU Batching. Los resultados validan matemáticamente las mejoras en eficiencia computacional y exponen los límites empíricos del modelo actual.

---

## 1. Rendimiento Global en la Intersección
La siguiente tabla muestra el rendimiento de las heurísticas calculando promedios **exclusivamente sobre los 19 tableros que todas las heurísticas lograron resolver**. Esto elimina el sesgo de supervivencia.

| Heurística | Resueltos (/40) | Nodos Promedio (Intersección) | Tiempo Promedio (Intersección) |
| :--- | :--- | :--- | :--- |
| **Manhattan** | 22 / 40 | 6,478.7 nodos | 62.5 ms |
| **Hungarian** | 37 / 40 | 2,331.2 nodos | 24.1 ms |
| **Neural Sequential** | 24 / 40 | 1,608.9 nodos | 9,925.5 ms |
| **Neural Batched Massive**| 33 / 40 | 2,135.9 nodos | 396.1 ms |

**Hallazgos Clave:**
- **Mejor calidad teórica de guía (Inviable en producción):** El modo secuencial (`neural_sequential`) logra un **31% de reducción** en nodos expandidos (1,608.9 vs 2,331.2) respecto al baseline `Hungarian`. Sin embargo, su latencia (~9,925 ms) lo hace impracticable.
- **Mejor opción real de producción (Viable):** El procesamiento por fronteras (`neural_batched_massive`) acelera el tiempo de búsqueda neuronal **25x** (396 ms vs 9.9 s), manteniendo una reducción real del **8.4% en nodos expandidos** respecto a Hungarian (2,135.9 vs 2,331.2). Esta es la configuración recomendada para despliegue.

---

## 2. Tasa de Resolución por Densidad (Box Count)
El análisis estratificado demuestra que la tasa de éxito de la red no es uniforme, sino que correlaciona de manera inversamente proporcional con la variable `box_count`.

| Cajas (Box Count) | Tableros Totales | Hungarian (Resolución) | Neural Masivo (Resolución) | Degradación Relativa |
| :--- | :--- | :--- | :--- | :--- |
| **1 caja** | 4 | 100% (4/4) | 100% (4/4) | 0% |
| **2 cajas** | 6 | 100% (6/6) | 100% (6/6) | 0% |
| **3 cajas** | 5 | 100% (5/5) | 100% (5/5) | 0% |
| **4 cajas** | 5 | 100% (5/5) | 100% (5/5) | 0% |
| **5 cajas** | 5 | 80% (4/5) | 80% (4/5) | 0% |
| **6 cajas** | 5 | 100% (5/5) | 80% (4/5) | -20% |
| **7 cajas** | 10 | 80% (8/10) | 50% (5/10) | -30% |

**Hallazgo Clave:** La tasa de resolución por `box_count` confirma en inferencia real (A*) el patrón que documentamos durante el entrenamiento: el modelo sufre de un **colapso de ranking intra-bucket** fuertemente correlacionado con la densidad de cajas, no con los pushes. La red se degrada dramáticamente a partir de las 6 cajas (cayendo al 50% en 7 cajas frente al 80% del Húngaro).

---

## 3. Análisis de Fallos Extremos (Hungarian vs Neural)
Aislamos los 5 tableros específicos que el Húngaro resolvió pero donde `neural_batched_massive` fracasó.

| Board ID | Cajas | Dificultad (Bucket) | Nodos Húngaro | Nodos Neural (al fallar) | Diagnóstico |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **27** | 5 | 71-90 | 40,505 n | 111,972 n | Explosión de Nodos |
| **31** | 7 | 91-100 (95 p) | 23,279 n | 60,617 n | Explosión de Nodos |
| **33** | 7 | 91-100 | 8,248 n | 59,981 n | Explosión de Nodos |
| **34** | 7 | 91-100 | 4,475 n | 64,233 n | Explosión de Nodos |
| **36** | 6 | 101+ (105 p) | 151,548 n | 167,112 n | Explosión de Nodos |

**Conclusión Mecánica (Límite Físico Confirmado):**
En `include/solver_template.h`, el límite de A* está configurado a `max_seconds = 120.0` o `max_nodes = 500,000` (nodos *generados*, equivalente a ~150,000 nodos *expandidos*). 
Ninguno de estos tableros falló por *timeout de tiempo* (lentitud). En el tablero 36, la GPU generó 500,000 hijos en apenas **26 segundos**, chocando contra el límite de memoria/nodos del árbol. Esto confirma categóricamente que el fallo es una **explosión de nodos**: la red pierde fiabilidad de ranking en cajas altas (colapso intra-bucket), subestimando ramas muertas y expandiendo el árbol hasta agotar el límite estructural.

---

## 4. Optimalidad de la Solución (Speed vs Optimality Trade-off)
Evaluamos empíricamente la calidad de la solución encontrada por la heurística neuronal (`neural_batched_massive`) frente al baseline matemático (`Hungarian`), medido sobre la intersección estricta de 19 tableros resueltos por ambos. Dado que el regresor neuronal no garantiza admisibilidad estricta, existe un riesgo de encontrar rutas subóptimas.

| Métrica de Optimalidad | Frecuencia | Porcentaje |
| :--- | :--- | :--- |
| **Igual de Óptima** (Mismos pushes) | 18 tableros | 94.7% |
| **Subóptima** (Más pushes) | 1 tablero | 5.3% |
| **Superóptima** (Menos pushes) | 0 tableros | 0.0% |

**Desglose del caso subóptimo:**
- **Board 14** (Cajas: 5): Hungarian encontró la solución en 28 pushes. Neural Massive la encontró en 30 pushes (+2 empujes adicionales).

**Hallazgo Clave:** A pesar de carecer de garantías matemáticas de admisibilidad absoluta, la red neuronal empíricamente preserva la **optimalidad perfecta en el 94.7% de los casos**. El único caso de degradación (+2 pushes) ocurrió justamente en el umbral de las 5 cajas, lo que vuelve a correlacionar la pérdida de precisión con la densidad, alineándose con los hallazgos de resolución y explosión de nodos.

---

## 5. Análisis Exploratorio de Densidad Estructural (KNN)
Para explorar si los 5 tableros fallidos (27, 31, 33, 34, 36) experimentaron errores por falta de volumen general de datos o por pertenecer a un "perfil estructural" atípico, realizamos un mapeo exploratorio de K-Vecinos Más Cercanos (KNN, $k=50$) contra una muestra de $\sim$9000 tableros del dataset de entrenamiento. Comparamos la distancia espacial de los tableros fallidos versus los tableros exitosos de su mismo *box count*, midiendo características morfológicas (dispersión de cajas, densidad de muros) y el "Gap" de dificultad (*Pushes Reales - Cota Húngara*).

**Resultados del Espacio Latente Estructural (Muestra Pequeña):**
- **5 Cajas (n=1 fallido vs n=3 exitosos):** El único tablero fallido (Board 27) muestra una distancia promedio a sus 50 vecinos más cercanos de 2.41, superando el máximo de los exitosos (1.91), y presenta un *Gap* masivo (40 empujes). Esta observación aislada sugiere que podría tratarse de un *outlier* estructural.
- **6 Cajas (n=1 fallido vs n=4 exitosos):** El único tablero fallido (Board 36) muestra un patrón similar, con una distancia extrema de 4.26 (vs 3.29 máximo en exitosos) y un *Gap* inusualmente alto (58 empujes). Al igual que en 5 cajas, esto apunta a una posible falta de representación local para perfiles muy atípicos.
- **7 Cajas (n=3 fallidos vs n=5 exitosos):** En este estrato, los tableros fallidos 33 y 34 presentan distancias al dataset de entrenamiento (1.88 y 1.47) completamente normales comparadas con los tableros exitosos (promedio 1.81). Tienen además *Gaps* bajos (2 y 6). En 2 de 3 tableros fallidos de 7 cajas, la falla no parece explicarse por escasez local de datos similares, sugiriendo que en este régimen el problema podría no ser únicamente de volumen de entrenamiento.

**Conclusión Exploratoria:** 
Dada la escasez de fallos totales ($n=5$) en el benchmark de 40 tableros, estos resultados no son estadísticamente concluyentes. Sin embargo, sugieren dos posibles mecanismos de falla distintos: en 5-6 cajas, los errores podrían deberse a escasez de datos locales (*outliers* de alto Gap solucionables con minería dirigida); mientras que en 7 cajas, el error en tableros "fáciles" y bien representados insinúa una posible saturación de la capacidad del modelo. Se requeriría ampliar el benchmark a una muestra significativamente mayor (100-200 tableros) para confirmar estadísticamente estas hipótesis.

---

## Recomendación Final de Arquitectura (Production Switch)
Basado en esta evidencia exhaustiva, la arquitectura en producción debe implementar un enrutador híbrido:
1. Si `box_count <= 5`: Usar **Neural Batched Massive** (mantiene alta tasa de resolución y gana en nodos).
2. Si `box_count >= 6`: Habilitar *fallback* puro a **Hungarian** (la red sufre colapso de ranking; la heurística clásica es más confiable).

Esta regla ya ha sido integrada de forma nativa en `src/neural_heuristic.cpp`.
