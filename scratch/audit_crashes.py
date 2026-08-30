import os
import glob
import pandas as pd
import re

LOG_DIR = "final_canonical_campaign"

def audit_logs():
    if not os.path.exists(LOG_DIR):
        print(f"Error: Directory {LOG_DIR} not found. Are you in the right path on the GTX4?")
        return

    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    if not log_files:
        print(f"No .txt logs found in {LOG_DIR}.")
        return

    results = []
    total_crashes = 0

    for log_path in log_files:
        filename = os.path.basename(log_path)
        # Parse filename: e.g., full_surrogate_shell1_seed44_cores24.txt
        parts = filename.replace(".txt", "").split("_shell")
        heuristic = parts[0]
        rest = parts[1].split("_seed")
        shell = rest[0]
        seed_cores = rest[1].split("_cores")
        seed = seed_cores[0]

        has_fallback = False
        has_circuit_breaker = False
        termination_reason = "UNKNOWN"
        time_s = "UNKNOWN"
        pushes = "UNKNOWN"

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
            # Check for crashes/fallbacks
            if "Fallback Silencioso" in content or "Connection refused" in content or "Timeout" in content:
                has_fallback = True
            if "circuit breaker" in content.lower() or "abortando evolucion" in content.lower():
                has_circuit_breaker = True
                
            # Find termination reason
            if "Termination reason: STAGNATION" in content or "STAGNATION" in content:
                termination_reason = "STAGNATION"
            elif "Termination reason: MAX_EVALUATIONS" in content:
                termination_reason = "MAX_EVALS"
            elif "Termination reason: TIME_LIMIT" in content:
                termination_reason = "TIME_LIMIT"
            elif has_circuit_breaker:
                termination_reason = "CRASH_CIRCUIT_BREAKER"
                
            # Find time and pushes
            time_match = re.search(r"Total time: ([\d\.]+)s", content)
            if time_match:
                time_s = time_match.group(1)
            
            pushes_match = re.search(r"Top 5 Pushes.*?\n.*?(\d+)", content, re.IGNORECASE)
            if pushes_match:
                pushes = pushes_match.group(1)

        crashed = has_fallback or has_circuit_breaker
        if crashed:
            total_crashes += 1

        results.append({
            "Variant": heuristic,
            "Shell": shell,
            "Seed": seed,
            "Time_s": time_s,
            "Pushes": pushes,
            "Termination": termination_reason,
            "Had_Fallbacks": has_fallback
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Variant", "Shell", "Seed"])
    
    print("\n=== AUDITORIA DE CRASHES EN DATASET CANONICO ===")
    print(df.to_string(index=False))
    
    print("\nResumen de Fallos:")
    print(f"Corridas con Fallbacks/Crashes: {total_crashes} de {len(log_files)}")
    print(df["Termination"].value_counts())
    
    # Específico para Seed 44
    seed44 = df[(df["Shell"] == "1") & (df["Seed"] == "44")]
    if not seed44.empty:
        print("\n[!] DETALLE SEMILLA 44 (Shell 1):")
        print(seed44.to_string(index=False))

if __name__ == "__main__":
    audit_logs()
