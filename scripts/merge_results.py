#!/usr/bin/env python3
"""
merge_results.py
================
Combina los CSVs parciales (uno por algoritmo) en un único
archivo exp1_raw_data.csv listo para el análisis.

Uso:
    python3 scripts/merge_results.py
    python3 scripts/merge_results.py --out mi_resultado.csv
"""

import os, sys, argparse
import pandas as pd

PARTIAL_FILES = [
    "exp1_raw_data_GA.csv",
    "exp1_raw_data_ES.csv",
    "exp1_raw_data_SA.csv",
]
OUTPUT_DEFAULT = "exp1_raw_data.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=OUTPUT_DEFAULT,
                        help="Archivo de salida combinado")
    args = parser.parse_args()

    dfs = []
    for f in PARTIAL_FILES:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"[OK] {f}: {len(df)} filas  ({df['Algoritmo'].unique()})")
            dfs.append(df)
        else:
            print(f"[WARN] No encontrado: {f}  (se omitirá)")

    if not dfs:
        print("[ERROR] No se encontró ningún archivo parcial.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"\n[OK] Archivos combinados → {args.out}  ({len(combined)} filas totales)")

    # Resumen rápido
    print("\nResumen por algoritmo y FO:")
    print(combined.groupby(['Algoritmo', 'FO'])['Fitness'].describe()[['count','mean','median','max']].to_string())

if __name__ == "__main__":
    main()
