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
    
    # Extraeremos el log específico que pidió Claude
    target_log = "full_surrogate_shell5_seed43_cores24.txt"
    target_log_content = None

    for log_path in log_files:
        filename = os.path.basename(log_path)
        if filename == target_log:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                target_log_content = f.read()

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
                termination_reason = "TIME_LIMIT_CPP"
            
            # Si el C++ no reportó término, investigar por qué (Timeout de Python, Segfault, etc)
            if termination_reason == "NO_REPORTADO":
                if "Fallback Silencioso a A*" in content:
                    termination_reason = "ABORTED_DURING_FALLBACK?"
                else:
                    termination_reason = "PYTHON_TIMEOUT_OR_SEGFAULT"

            # Buscar pushes
            pushes = "UNKNOWN"
            # Buscamos la mejor aptitud histórica guardada en el log si no llegó al final
            best_fitness_match = re.findall(r"BEST ([\d\.]+)", content)
            if best_fitness_match:
                pushes = best_fitness_match[-1] # el último reportado
            
            # Buscar tiempo (si no está al final, asume timeout de Python)
            time_s = "380.0 (Timeout)"
            time_match = re.search(r"Total time: ([\d\.]+)s", content)
            if time_match:
                time_s = time_match.group(1)

        results.append({
            "Variant": heuristic,
            "Shell": shell,
            "Seed": seed,
            "Time_s": time_s,
            "Pushes": pushes,
            "Termination": termination_reason
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Variant", "Shell", "Seed"])
    
    print("=== AUDITORÍA CORREGIDA ===")
    anomalous = df[df["Termination"].isin(["NO_REPORTADO", "PYTHON_TIMEOUT_OR_SEGFAULT", "ABORTED_DURING_FALLBACK?"])]
    print(f"\nCorridas sin terminación limpia de C++: {len(anomalous)}")
    if len(anomalous) > 0:
        print(anomalous.to_string(index=False))
        
    print("\nResumen Total:")
    print(df["Termination"].value_counts())
    
    print("\n=== LOG SOLICITADO POR CLAUDE (ÚLTIMAS 25 LÍNEAS) ===")
    print(f"Archivo: {target_log}")
    if target_log_content:
        lines = target_log_content.split('\n')
        print("\n".join(lines[-25:]))
    else:
        print("No se encontró el archivo.")

if __name__ == "__main__":
    analyze_logs()
