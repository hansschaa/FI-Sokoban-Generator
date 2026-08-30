import os
import glob
import pandas as pd
import re
import textwrap

LOG_DIR = "final_canonical_campaign"

def analyze_logs():
    if not os.path.exists(LOG_DIR):
        print(f"Error: No encuentro el directorio {LOG_DIR}")
        return

    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    results = []

    for log_path in log_files:
        filename = os.path.basename(log_path)
        parts = filename.replace(".txt", "").split("_shell")
        heuristic = parts[0]
        rest = parts[1].split("_seed")
        shell = rest[0]
        seed_cores = rest[1].split("_cores")
        seed = seed_cores[0]

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
            # Buscar el verdadero motivo de término reportado por C++
            termination_reason = "NO_REPORTADO"
            if "Criterio de Parada Alcanzado: STAGNATION" in content:
                termination_reason = "STAGNATION"
            elif "Criterio de Parada Alcanzado: MAX_EVALUATIONS" in content:
                termination_reason = "MAX_EVALS"
            elif "Criterio de Parada Alcanzado: TIME LIMIT" in content:
                termination_reason = "TIME_LIMIT"
            
            # 1. VERIFICACIÓN ESTRUCTURAL ESTRICTA
            # Confirmar que la corrida realmente terminó y no fue matada a la fuerza
            has_stats_block = "[ES STATS]" in content
            has_final_pop = "[TOP_FINAL_POPULATION]" in content
            has_exception = "terminate called after throwing an instance" in content or "Segmentation fault" in content
            
            structural_integrity = "CLEAN"
            if has_exception:
                structural_integrity = "EXCEPTION/SEGFAULT"
                termination_reason = "CRASH_REAL"
            elif not has_stats_block or not has_final_pop:
                structural_integrity = "TRUNCATED (Timeout/OOM?)"
                termination_reason = "CRASH_TRUNCATED"

            results.append({
                "Variant": heuristic,
                "Shell": int(shell),
                "Seed": int(seed),
                "Termination": termination_reason,
                "Integrity": structural_integrity
            })

    df = pd.DataFrame(results)
    
    print("=== VERIFICACIÓN ESTRUCTURAL ===")
    anomalous = df[df["Integrity"] != "CLEAN"]
    print(f"Corridas truncadas o con excepciones: {len(anomalous)}")
    if len(anomalous) > 0:
        print(anomalous.to_string(index=False))
        
    print("\n=== MATRIZ DE TERMINACIONES (Shell 1 vs Shell 5) ===")
    
    # Filtrar solo Shell 1 y 5 para la matriz
    df_matrix = df[df["Shell"].isin([1, 5])]
    
    # Crear tabla pivote: Variant -> [Shell 1 TIME_LIMIT] [Shell 5 TIME_LIMIT]
    pivot = df_matrix.pivot_table(
        index="Variant",
        columns=["Shell"],
        values="Termination",
        aggfunc=lambda x: f"{(x == 'TIME_LIMIT').sum()}/{len(x)}"
    )
    
    # Renombrar columnas
    pivot.columns = [f"Shell {c} (TIME_LIMIT)" for c in pivot.columns]
    print(pivot)
    
    print("\n=== RESUMEN GLOBAL (250 CORRIDAS) ===")
    print(df["Termination"].value_counts())

if __name__ == "__main__":
    analyze_logs()
