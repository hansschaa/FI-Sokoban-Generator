import json

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.rstrip("\n").split("\n")]
    }

def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")]
    }

cells = []

# ── INTRODUCCIÓN ──────────────────────────────────────────────────────────────
cells.append(markdown_cell("""# Análisis del Experimento: Generación Procedural de Niveles de Sokoban

Este notebook procesa `exp1_raw_data.csv` y responde a las dos preguntas de investigación del artículo:

* **RQ1:** ¿Qué metaheurística logra el mejor rendimiento agregado en todo el conjunto de funciones objetivas y en qué medida esta jerarquía depende del paisaje?
* **RQ2:** Which metaheuristic demonstrates the highest robustness to the choice of objective function, minimizing intra-algorithm variance across disparate fitness landscapes?
* **Adicional:** Diversidad Estructural (tableros únicos) y Costo Computacional (segundos/run).

**Diseño:** 3 Algoritmos × 3 Funciones Objetivo × 10 Tableros × 15 Semillas = 1350 ejecuciones."""))

# ── SETUP ─────────────────────────────────────────────────────────────────────
cells.append(markdown_cell("## 0. Setup y Carga de Datos"))
cells.append(code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("[WARN] scikit_posthocs no disponible. Instalar con: pip install scikit-posthocs")

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"SA": "#4C72B0", "ES": "#DD8452", "GA": "#55A868"}

df = pd.read_csv("exp1_raw_data.csv")
df = df.rename(columns={
    "Fitness_Bruto": "Fitness",
    "BT_id": "Shell_ID",
    "Semilla_ID": "Rep",
    "Tiempo_Segundos": "Time_s"
})
df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")
df = df.dropna(subset=["Fitness"])

# Convertir a positivo: los algoritmos internamente minimizan (fitness negativo).
# Para coherencia visual, trabajamos con el valor absoluto.
# Mayor valor absoluto = mayor complejidad/calidad del nivel generado.
df["Fitness"] = df["Fitness"].abs()

print(f"Filas totales: {len(df)}")
print(f"Algoritmos: {sorted(df['Algoritmo'].unique())}")
print(f"FOs: {sorted(df['FO'].unique())}")
print(f"Tableros: {df['Shell_ID'].nunique()}")
print(f"Repeticiones: {df['Rep'].nunique()}")
print(f"Fitness min/max tras abs(): {df['Fitness'].min():.1f} / {df['Fitness'].max():.1f}")
df.head()"""))

# ── NORMALIZACIÓN ──────────────────────────────────────────────────────────────
cells.append(markdown_cell("""## 1. Normalización Min-Max del Fitness

Los fitness ya fueron convertidos a positivos (valor absoluto) en la celda anterior.
Dado que las 3 FOs operan en escalas distintas (ej. Pushes puede superar 100 empujes,
mientras Deadlocks puede quedarse en 20), aplicamos **normalización Min-Max por grupo (FO, Shell_ID)**
para hacer las comparaciones inter-FO justas.

Resultado: 1.0 = mejor resultado en ese tablero+FO, 0.0 = peor. Siempre: mayor = mejor.
⚠️ El ranking dentro de cada grupo es relativo a los 3 algoritmos que compiten en él."""))
cells.append(code_cell("""df_norm = df.copy()
df_norm["Fitness_norm"] = np.nan

for (fo, sid), grp in df.groupby(["FO", "Shell_ID"]):
    mn, mx = grp["Fitness"].min(), grp["Fitness"].max()
    if mx > mn:
        df_norm.loc[grp.index, "Fitness_norm"] = (grp["Fitness"] - mn) / (mx - mn)
    else:
        # Todos los algoritmos son iguales en este grupo: asignar 0.5 neutral
        df_norm.loc[grp.index, "Fitness_norm"] = 0.5

print("Fitness_norm - estadísticas globales:")
print(df_norm.groupby("Algoritmo")["Fitness_norm"].describe().round(4))"""))

# ── RQ1 ────────────────────────────────────────────────────────────────────────
cells.append(markdown_cell("""## 2. RQ1 — Rendimiento Agregado

### 2a. Ranking Global (todas las FOs juntas)
¿Qué algoritmo obtiene el mayor fitness normalizado promedio en el conjunto completo de landscapes?"""))
cells.append(code_cell("""agg = (df_norm.groupby("Algoritmo")["Fitness_norm"]
       .agg(["mean","std","median"])
       .rename(columns={"mean":"Mean","std":"Std","median":"Median"})
       .sort_values("Mean", ascending=False))
print("Ranking global (mayor = mejor rendimiento agregado):")
display(agg.round(4))

fig, ax = plt.subplots(figsize=(7, 5))
order = agg.index.tolist()
sns.boxplot(data=df_norm, x="Algoritmo", y="Fitness_norm",
            order=order, palette=PALETTE, ax=ax)
ax.set_title("RQ1 — Distribución Fitness Normalizado (Global)")
ax.set_ylabel("Fitness Normalizado [0-1]  (mayor = mejor)")
ax.set_xlabel("Metaheurística")
plt.tight_layout()
plt.show()"""))

cells.append(markdown_cell("""### 2b. Jerarquía por Función Objetivo (¿depende del paisaje?)

Parte de la RQ1 pregunta **en qué medida el ranking cambia según el landscape**.
Si el orden SA > ES > GA se mantiene igual en las 3 FOs, la jerarquía es estable.
Si cambia, el rendimiento depende del paisaje."""))
cells.append(code_cell("""# Media normalizada por (Algoritmo, FO)
by_fo = (df_norm.groupby(["FO", "Algoritmo"])["Fitness_norm"]
         .mean()
         .unstack("Algoritmo"))
print("Media Fitness Normalizado por FO × Algoritmo:")
display(by_fo.round(4))

# Heatmap
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(by_fo, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, ax=ax)
ax.set_title("RQ1 — Rendimiento por Función Objetivo × Algoritmo")
ax.set_ylabel("Función Objetivo")
ax.set_xlabel("Algoritmo")
plt.tight_layout()
plt.show()

# Boxplot por FO separado
g = sns.FacetGrid(df_norm, col="FO", height=4, aspect=0.9, sharey=True)
g.map_dataframe(sns.boxplot, x="Algoritmo", y="Fitness_norm",
                order=["SA","ES","GA"], palette=PALETTE)
g.set_titles("{col_name}")
g.set_axis_labels("Algoritmo", "Fitness Normalizado")
g.figure.suptitle("RQ1 — Distribución por FO (¿depende el ranking del paisaje?)", y=1.02)
plt.tight_layout()
plt.show()"""))

cells.append(markdown_cell("""### 2c. Test de Friedman + Post-hoc Wilcoxon (Bonferroni)

Usamos **Test de Friedman** (no Kruskal-Wallis) porque las observaciones **no son independientes**:
el mismo tablero y la misma semilla son evaluados por los 3 algoritmos. Friedman bloquea por (Shell_ID, FO, Rep)."""))
cells.append(code_cell("""pivot = df_norm.pivot_table(
    index=["Shell_ID", "FO", "Rep"],
    columns="Algoritmo",
    values="Fitness_norm"
).dropna()

print(f"Bloques completos para Friedman: {len(pivot)}")

if len(pivot) >= 3:
    groups = [pivot[col].values for col in pivot.columns]
    stat, p = stats.friedmanchisquare(*groups)
    print(f"\\nTest de Friedman: chi2={stat:.4f}  p-value={p:.4e}")
    if p < 0.05:
        print("✅ Diferencias estadísticamente significativas (p < 0.05)")
    else:
        print("❌ Sin diferencias significativas")

    # Post-hoc: Wilcoxon por pares con corrección Bonferroni
    print("\\nPost-hoc: Wilcoxon por pares (corrección Bonferroni):")
    algos = pivot.columns.tolist()
    pairs = list(combinations(algos, 2))
    alpha = 0.05 / len(pairs)  # Bonferroni
    for a1, a2 in pairs:
        s, pval = stats.wilcoxon(pivot[a1], pivot[a2])
        sig = "✅ SIG" if pval < alpha else "❌ NS"
        print(f"  {a1} vs {a2}: W={s:.1f}, p={pval:.4e}  {sig} (α_Bonferroni={alpha:.4f})")"""))

# ── RQ2 ────────────────────────────────────────────────────────────────────────
cells.append(markdown_cell("""## 3. RQ2 — Robustez (Varianza Intra-Algoritmo entre Paisajes)

Un algoritmo **robusto** mantiene rendimiento consistente sin importar cuál FO se use.
Medimos robustez con el **Coeficiente de Variación (CV = σ/μ)** de la media del fitness normalizado entre las 3 FOs.
Un CV menor indica menor sensibilidad al cambio de landscape, es decir mayor robustez."""))
cells.append(code_cell("""def coef_variation(series):
    m = series.mean()
    if m == 0 or np.isnan(m):
        return np.nan
    return series.std() / m

# Media por (Algoritmo, FO) — solo 3 puntos por algoritmo
mean_per_fo = (df_norm.groupby(["Algoritmo", "FO"])["Fitness_norm"]
               .mean().reset_index())

print("Media Fitness Normalizado por Algoritmo × FO:")
display(mean_per_fo.pivot(index="Algoritmo", columns="FO", values="Fitness_norm").round(4))

cv_table = (mean_per_fo.groupby("Algoritmo")["Fitness_norm"]
            .apply(coef_variation)
            .sort_values()
            .to_frame("CV (↓ = más robusto)"))

print("\\nCoeficiente de Variación entre FOs (menor = más robusto):")
display(cv_table.round(4))

# Barplot CV
fig, ax = plt.subplots(figsize=(6, 4))
colors = [PALETTE[a] for a in cv_table.index]
cv_table["CV (↓ = más robusto)"].plot(kind="bar", ax=ax, color=colors, edgecolor="black")
ax.set_title("RQ2 — Robustez: CV entre Funciones Objetivo")
ax.set_ylabel("Coeficiente de Variación (CV)")
ax.set_xlabel("Metaheurística")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()"""))

cells.append(markdown_cell("""### 3b. Distribución de fitness por Algoritmo × FO — figura para RQ2

Para RQ2, la figura clave es **1 subplot por algoritmo** con las FOs en el eje X.
Así se ve directamente si las cajas de cada algoritmo se mantienen alineadas al cambiar de landscape (robusto)
o si se desplazan marcadamente (sensible al paisaje)."""))
cells.append(code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fo_order = sorted(df_norm["FO"].unique())
for ax, algo in zip(axes, ["SA", "ES", "GA"]):
    sub = df_norm[df_norm["Algoritmo"] == algo]
    sns.boxplot(data=sub, x="FO", y="Fitness_norm",
                order=fo_order, color=PALETTE[algo], ax=ax)
    ax.set_title(algo, fontsize=13, fontweight="bold")
    ax.set_xlabel("Función Objetivo")
    ax.set_ylabel("Fitness Normalizado [0-1]" if ax == axes[0] else "")
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("RQ2 — Robustez: Distribución por Algoritmo × Landscape", y=1.02)
plt.tight_layout()
plt.show()"""))

# ── DIVERSIDAD ─────────────────────────────────────────────────────────────────
cells.append(markdown_cell("""## 4. Diversidad Estructural y Costo Computacional

**Diversidad** = número de tableros con hash único por grupo (Algoritmo, FO, Shell_ID), de un máximo de 15 (semillas).
⚠️ El hash detecta duplicados **exactos**; tableros casi idénticos cuentan como distintos.

**Tiempo** = segundos promedio por ejecución."""))
cells.append(code_cell("""# Diversidad: hashes únicos por (Algoritmo, FO, Shell)
diversity = (df.groupby(["Algoritmo","FO","Shell_ID"])["Board_Hash"]
             .nunique().reset_index(name="Unique_Boards"))

# Media de diversidad por (Algoritmo, FO)
div_summary = (diversity.groupby(["Algoritmo","FO"])["Unique_Boards"]
               .mean().unstack("FO"))
print("Diversidad Promedio de Tableros Únicos (max = 15):")
display(div_summary.round(2))

# Tiempo promedio por (Algoritmo, FO)
time_summary = (df.groupby(["Algoritmo","FO"])["Time_s"]
                .mean().unstack("FO"))
print("\\nTiempo Promedio por Ejecución (segundos):")
display(time_summary.round(2))

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

div_agg = diversity.groupby("Algoritmo")["Unique_Boards"].mean().sort_values(ascending=False)
div_agg.plot(kind="bar", ax=axes[0],
             color=[PALETTE[a] for a in div_agg.index], edgecolor="black")
axes[0].set_title("Diversidad Promedio Agregada (Max = 15)")
axes[0].set_ylabel("Tableros Únicos Generados")
axes[0].tick_params(axis="x", rotation=0)
axes[0].axhline(y=15, color="red", linestyle="--", alpha=0.5, label="Max=15")
axes[0].legend()

time_agg = df.groupby("Algoritmo")["Time_s"].mean().sort_values()
time_agg.plot(kind="bar", ax=axes[1],
              color=[PALETTE[a] for a in time_agg.index], edgecolor="black")
axes[1].set_title("Costo Computacional Promedio")
axes[1].set_ylabel("Segundos por Ejecución")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()"""))

# ── RESUMEN ────────────────────────────────────────────────────────────────────
cells.append(markdown_cell("## 5. Resumen Final de Resultados"))
cells.append(code_cell("""print("=" * 60)
print("RESUMEN DE RESULTADOS")
print("=" * 60)

print("\\n[RQ1] Ranking global de rendimiento (Fitness Normalizado):")
display(agg.round(4))

print("\\n[RQ1] ¿La jerarquía depende del paisaje? (Ranking por FO):")
display(by_fo.round(4))

print("\\n[RQ2] Robustez (CV entre FOs — menor es más robusto):")
display(cv_table.round(4))

print("\\n[DIVERSIDAD] Tableros únicos promedio por algoritmo:")
display(div_agg.round(2).to_frame("Unique_Boards_avg"))

print("\\n[TIEMPO] Segundos promedio por ejecución:")
display(time_agg.round(2).to_frame("Time_s_avg"))"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Analysis_RQ1_RQ2.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook escrito correctamente.")
