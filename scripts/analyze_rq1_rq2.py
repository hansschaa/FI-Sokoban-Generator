#!/usr/bin/env python3
"""
analyze_rq1_rq2.py
==================
Análisis estadístico de los resultados del experimento.

Responde:
  RQ1 — ¿Qué metaheurística logra el mejor rendimiento agregado
         sobre todas las funciones objetivo?
  RQ2 — ¿Qué metaheurística es más robusta (menor varianza interna
         al cambiar la función objetivo)?

USO:
    python3 scripts/analyze_rq1_rq2.py --input exp1_raw_data.csv

REQUIERE:
    pip install pandas numpy scipy matplotlib seaborn scikit-posthocs
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin display; guarda como PNG
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# scikit-posthocs opcional (Wilcoxon post-hoc)
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("[INFO] scikit-posthocs no disponible. Se omite Wilcoxon post-hoc.")
    print("       pip install scikit-posthocs")


OUTPUT_DIR = "results_rq1_rq2"


# ============================================================
# HELPERS
# ============================================================

def coef_variation(series):
    """Coeficiente de variación (CV = std / mean)."""
    m = series.mean()
    return (series.std() / m) if m != 0 else float("nan")


def normalize_per_fo(df):
    """
    Normalización Min-Max por (FO, Shell_ID) para hacer los fitness
    comparables entre distintas funciones objetivo.
    Retorna la columna 'Fitness_norm'.
    """
    df = df.copy()
    df["Fitness_norm"] = float("nan")
    for (fo, sid), grp in df.groupby(["FO", "Shell_ID"]):
        mn, mx = grp["Fitness"].min(), grp["Fitness"].max()
        if mx > mn:
            df.loc[grp.index, "Fitness_norm"] = (grp["Fitness"] - mn) / (mx - mn)
        else:
            df.loc[grp.index, "Fitness_norm"] = 0.5
    return df


# ============================================================
# RQ1 — RENDIMIENTO AGREGADO
# ============================================================

def analyze_rq1(df, out_dir):
    print("\n" + "="*60)
    print("  RQ1: Rendimiento Agregado por Metaheurística")
    print("="*60)

    df_norm = normalize_per_fo(df)

    # Media de Fitness normalizado por algoritmo (todas las FOs)
    agg = (df_norm.groupby("Algoritmo")["Fitness_norm"]
                  .agg(["mean", "std", "median"])
                  .rename(columns={"mean": "Mean_norm",
                                   "std":  "Std_norm",
                                   "median": "Median_norm"})
                  .sort_values("Mean_norm", ascending=False))
    print("\nRanking agregado (fitness normalizado, mayor = mejor):")
    print(agg.to_string())
    agg.to_csv(os.path.join(out_dir, "rq1_ranking_agregado.csv"))

    # Media por (Algoritmo, FO) — para ver landscape-dependency
    by_fo = (df_norm.groupby(["Algoritmo", "FO"])["Fitness_norm"]
                    .mean()
                    .unstack("FO"))
    print("\nMedia normalizada por Algoritmo × FO:")
    print(by_fo.to_string())
    by_fo.to_csv(os.path.join(out_dir, "rq1_mean_by_fo.csv"))

    # Heatmap Algoritmo × FO
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(by_fo, annot=True, fmt=".3f", cmap="YlGnBu",
                linewidths=0.5, ax=ax)
    ax.set_title("RQ1 — Media Fitness Normalizado por Algoritmo × FO")
    ax.set_ylabel("Algoritmo")
    ax.set_xlabel("Función Objetivo")
    plt.tight_layout()
    path = os.path.join(out_dir, "rq1_heatmap_algo_fo.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[Figura] {path}")

    # Boxplot por algoritmo (fitness normalizado agregado)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = agg.index.tolist()
    sns.boxplot(data=df_norm, x="Algoritmo", y="Fitness_norm",
                order=order, palette="Set2", ax=ax)
    ax.set_title("RQ1 — Distribución Fitness Normalizado (todas las FOs)")
    ax.set_xlabel("Metaheurística")
    ax.set_ylabel("Fitness normalizado [0-1]")
    plt.tight_layout()
    path = os.path.join(out_dir, "rq1_boxplot_agregado.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Figura] {path}")

    # Test de Friedman — cada (Shell_ID, FO, Rep) como bloque
    # Pivotamos: filas = observación (Shell_ID, FO, Rep), columnas = Algoritmo
    pivot = df_norm.pivot_table(
        index=["Shell_ID", "FO", "Rep"],
        columns="Algoritmo",
        values="Fitness_norm"
    ).dropna()

    if len(pivot) < 3:
        print("[WARN] Datos insuficientes para el test de Friedman.")
    else:
        groups = [pivot[col].values for col in pivot.columns]
        stat, p = stats.friedmanchisquare(*groups)
        result_str = (
            f"Test de Friedman: chi2={stat:.4f}  p={p:.6f}\n"
            + ("✅ Diferencias significativas (p < 0.05)" if p < 0.05
               else "❌ Sin diferencias significativas")
        )
        print(f"\n{result_str}")

        with open(os.path.join(out_dir, "rq1_friedman.txt"), "w") as f:
            f.write(result_str + "\n")

        # Wilcoxon post-hoc (pares de algoritmos sobre el pivot)
        if HAS_POSTHOCS and p < 0.05:
            df_long = pivot.reset_index().melt(
                id_vars=["Shell_ID", "FO", "Rep"],
                var_name="Algoritmo",
                value_name="Fitness_norm"
            )
            posthoc = sp.posthoc_wilcoxon(
                df_long, val_col="Fitness_norm", group_col="Algoritmo",
                p_adjust="bonferroni"
            )
            print("\nWilcoxon post-hoc (Bonferroni):")
            print(posthoc.to_string())
            posthoc.to_csv(os.path.join(out_dir, "rq1_posthoc_wilcoxon.csv"))


# ============================================================
# RQ2 — ROBUSTEZ (VARIANZA INTRA-ALGORITMO)
# ============================================================

def analyze_rq2(df, out_dir):
    print("\n" + "="*60)
    print("  RQ2: Robustez — Varianza Intra-Algoritmo entre FOs")
    print("="*60)

    df_norm = normalize_per_fo(df)

    # CV por algoritmo entre FOs (usando media por algo×fo primero)
    mean_per_fo = (df_norm.groupby(["Algoritmo", "FO"])["Fitness_norm"]
                          .mean()
                          .reset_index())

    cv_table = (mean_per_fo.groupby("Algoritmo")["Fitness_norm"]
                            .agg(coef_variation)
                            .rename("CV_entre_FOs")
                            .sort_values())
    print("\nCoeficiente de variación entre FOs (menor = más robusto):")
    print(cv_table.to_string())
    cv_table.to_csv(os.path.join(out_dir, "rq2_cv_entre_fo.csv"))

    # Barplot de CV
    fig, ax = plt.subplots(figsize=(6, 4))
    cv_table.plot(kind="bar", ax=ax, color=["#4CAF50", "#FF9800", "#F44336"][:len(cv_table)],
                  edgecolor="black")
    ax.set_title("RQ2 — Robustez: CV entre Funciones Objetivo")
    ax.set_xlabel("Metaheurística")
    ax.set_ylabel("Coeficiente de Variación (↓ = más robusto)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    path = os.path.join(out_dir, "rq2_cv_barplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[Figura] {path}")

    # Boxplot por (Algoritmo, FO) para ver la dispersión visual
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, algo in zip(axes, sorted(df_norm["Algoritmo"].unique())):
        sub = df_norm[df_norm["Algoritmo"] == algo]
        sns.boxplot(data=sub, x="FO", y="Fitness_norm",
                    palette="Set3", ax=ax)
        ax.set_title(f"{algo}")
        ax.set_xlabel("Función Objetivo")
        ax.set_ylabel("Fitness norm." if ax == axes[0] else "")
    fig.suptitle("RQ2 — Fitness Normalizado por Algoritmo × FO", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "rq2_boxplot_por_fo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figura] {path}")

    # Tabla resumen completa
    # Usamos una función nombrada para que pandas la llame por su __name__
    # (evita diferencias entre versiones de pandas con lambdas anónimas)
    def _cv(s):
        return coef_variation(s)
    _cv.__name__ = "CV"

    summary = (df_norm.groupby(["Algoritmo", "FO"])["Fitness_norm"]
                      .agg(["mean", "std", "median", _cv])
                      .rename(columns={"mean": "Mean", "std": "Std",
                                       "median": "Median"}))
    print("\nResumen completo por Algoritmo × FO:")
    print(summary.to_string())
    summary.to_csv(os.path.join(out_dir, "rq2_summary_tabla.csv"))


# ============================================================
# DIVERSIDAD Y TIEMPOS (PCG Metrics)
# ============================================================

def analyze_diversity_and_time(df, out_dir):
    print("\n" + "="*60)
    print("  DIVERSIDAD Y TIEMPOS (Costo vs Diversidad Estructural)")
    print("="*60)

    if "Board_Hash" not in df.columns or "Elapsed_ms" not in df.columns:
        print("[WARN] Columnas de diversidad/tiempo no encontradas en el CSV.")
        return

    # 1. Diversidad: hashes únicos por (Algoritmo, FO, Shell_ID)
    diversity = df.groupby(["Algoritmo", "FO", "Shell_ID"])["Board_Hash"].nunique().reset_index()
    # Promedio de diversidad por (Algoritmo, FO) (Max = 30 si reps=30)
    mean_div = diversity.groupby(["Algoritmo", "FO"])["Board_Hash"].mean().unstack("FO")
    
    print("\nDiversidad promedio (Nº de niveles únicos generados de 30 posibles):")
    print(mean_div.to_string())
    mean_div.to_csv(os.path.join(out_dir, "pcg_diversity_mean.csv"))

    # 2. Tiempos: Elapsed_ms
    mean_time = df.groupby(["Algoritmo", "FO"])["Elapsed_ms"].mean().unstack("FO") / 1000.0  # a segundos
    print("\nTiempo de ejecución promedio (segundos):")
    print(mean_time.to_string())
    mean_time.to_csv(os.path.join(out_dir, "pcg_time_mean_seconds.csv"))

    # 3. Barplot Diversidad
    fig, ax = plt.subplots(figsize=(7, 4))
    mean_div_agg = mean_div.mean(axis=1).sort_values(ascending=False)
    mean_div_agg.plot(kind="bar", ax=ax, color="#2196F3", edgecolor="black")
    ax.set_title("Diversidad Promedio Agregada (Max = 30)")
    ax.set_ylabel("Niveles Únicos Generados")
    ax.set_xlabel("Metaheurística")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    path_div = os.path.join(out_dir, "pcg_diversity_barplot.png")
    plt.savefig(path_div, dpi=150)
    plt.close()

    # 4. Barplot Tiempos
    fig, ax = plt.subplots(figsize=(7, 4))
    mean_time_agg = mean_time.mean(axis=1).sort_values(ascending=True)
    mean_time_agg.plot(kind="bar", ax=ax, color="#F44336", edgecolor="black")
    ax.set_title("Costo Computacional Promedio")
    ax.set_ylabel("Segundos por Ejecución")
    ax.set_xlabel("Metaheurística")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    path_time = os.path.join(out_dir, "pcg_time_barplot.png")
    plt.savefig(path_time, dpi=150)
    plt.close()
    
    print(f"\n[Figuras] {path_div} | {path_time}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="exp1_raw_data.csv",
                        help="CSV generado por run_experiment_rq1_rq2.py")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Carpeta de salida para figuras y tablas")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] No existe el archivo: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input)
    df["Fitness"] = pd.to_numeric(df["Fitness"], errors="coerce")
    df = df.dropna(subset=["Fitness"])
    print(f"[OK] {len(df)} filas cargadas de {args.input}")
    print(f"     Algoritmos: {sorted(df['Algoritmo'].unique())}")
    print(f"     FOs:        {sorted(df['FO'].unique())}")
    print(f"     Tableros:   {df['Shell_ID'].nunique()}")

    analyze_rq1(df, args.output)
    analyze_rq2(df, args.output)
    analyze_diversity_and_time(df, args.output)

    print(f"\n✅  Análisis completo. Resultados en: {args.output}/")


if __name__ == "__main__":
    main()
