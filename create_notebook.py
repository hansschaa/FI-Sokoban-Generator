import json

def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.split("\\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in code.split("\\n")]
    }

notebook = {
    "cells": [
        markdown_cell("# Análisis del Experimento: Generación de Niveles de Sokoban\\nEste notebook procesa `exp1_raw_data.csv` y responde a:\\n* **RQ1**: Rendimiento agregado entre metaheurísticas.\\n* **RQ2**: Robustez (varianza intra-algoritmo entre funciones objetivo).\\n* **Diversidad y Tiempos**: Capacidad de exploración vs Costo."),
        code_cell("import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nimport seaborn as sns\\nfrom scipy import stats\\n\\nsns.set_theme(style='whitegrid')\\n\\n# Carga de datos\\ndf = pd.read_csv('exp1_raw_data.csv')\\ndf = df.rename(columns={'Fitness_Bruto': 'Fitness', 'BT_id': 'Shell_ID', 'Semilla_ID': 'Rep', 'Tiempo_Segundos': 'Time_s'})\\ndf.head()"),
        
        markdown_cell("## RQ1: Rendimiento Agregado\\nNormalizamos (Min-Max) el fitness por (FO, Shell) para comparar el rendimiento de los algoritmos globalmente."),
        code_cell("df_norm = df.copy()\\ndf_norm['Fitness_norm'] = np.nan\\n\\nfor (fo, sid), grp in df.groupby(['FO', 'Shell_ID']):\\n    mn, mx = grp['Fitness'].min(), grp['Fitness'].max()\\n    if mx > mn:\\n        df_norm.loc[grp.index, 'Fitness_norm'] = (grp['Fitness'] - mn) / (mx - mn)\\n    else:\\n        df_norm.loc[grp.index, 'Fitness_norm'] = 0.5\\n\\nplt.figure(figsize=(8, 5))\\nsns.boxplot(data=df_norm, x='Algoritmo', y='Fitness_norm', palette='Set2')\\nplt.title('RQ1: Distribución Fitness Normalizado Global')\\nplt.ylabel('Fitness Normalizado [0-1]')\\nplt.show()\\n\\nagg = df_norm.groupby('Algoritmo')['Fitness_norm'].agg(['mean', 'std', 'median']).sort_values('mean', ascending=False)\\ndisplay(agg)"),
        
        markdown_cell("### Test Estadístico (Kruskal-Wallis)\\nVerificamos diferencias significativas."),
        code_cell("groups = [df_norm[df_norm['Algoritmo'] == alg]['Fitness_norm'].dropna() for alg in df_norm['Algoritmo'].unique()]\\nstat, p = stats.kruskal(*groups)\\nprint(f'Kruskal-Wallis: H={stat:.4f}, p-value={p:.2e}')"),

        markdown_cell("## RQ2: Robustez de la Metaheurística\\nMedimos el Coeficiente de Variación (CV) entre Funciones Objetivo. Menor CV implica mayor robustez ante cambios en la FO."),
        code_cell("def coef_variation(series):\\n    m = series.mean()\\n    return (series.std() / m) if m != 0 else np.nan\\n\\nmean_per_fo = df_norm.groupby(['Algoritmo', 'FO'])['Fitness_norm'].mean().reset_index()\\ncv_table = mean_per_fo.groupby('Algoritmo')['Fitness_norm'].apply(coef_variation).sort_values().to_frame('CV (menor = más robusto)')\\ndisplay(cv_table)\\n\\ncv_table.plot(kind='bar', color=['#4CAF50', '#FF9800', '#F44336'], edgecolor='black', legend=False)\\nplt.title('RQ2: Coeficiente de Variación (CV)')\\nplt.xticks(rotation=0)\\nplt.show()\\n\\nplt.figure(figsize=(10, 5))\\nsns.boxplot(data=df_norm, x='Algoritmo', y='Fitness_norm', hue='FO', palette='Set3')\\nplt.title('Rendimiento por FO')\\nplt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\\nplt.show()"),

        markdown_cell("## Diversidad y Costo Computacional\\nCantidad de niveles únicos generados (por hash) y tiempo de ejecución."),
        code_cell("diversity = df.groupby(['Algoritmo', 'FO', 'Shell_ID'])['Board_Hash'].nunique().reset_index()\\nmean_div = diversity.groupby('Algoritmo')['Board_Hash'].mean().sort_values(ascending=False).to_frame('Tableros Únicos Promedio')\\nmean_time = df.groupby('Algoritmo')['Time_s'].mean().sort_values().to_frame('Tiempo Promedio (s)')\\n\\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\\nmean_div.plot(kind='bar', ax=axes[0], color='#2196F3', edgecolor='black', legend=False)\\naxes[0].set_title('Diversidad Promedio')\\nmean_time.plot(kind='bar', ax=axes[1], color='#F44336', edgecolor='black', legend=False)\\naxes[1].set_title('Costo Computacional')\\nplt.show()\\n\\ndisplay(mean_div)\\ndisplay(mean_time)")
    ],
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
