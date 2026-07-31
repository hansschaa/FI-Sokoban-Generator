import os
import glob
import re

log_files = glob.glob("*.log")
if not log_files:
    print("No se encontraron archivos .log en el directorio actual (ej. es_run.log).")
    exit()

print("="*80)
print("ANALISIS DE DISYUNTORES DE SEGURIDAD Y DILUCION (ES STATS)")
print("="*80)

for log_file in log_files:
    print(f"\n--- Analizando: {log_file} ---")
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    current_run = None
    stats = []
    
    for line in lines:
        if "Running" in line and "Seed" in line:
            current_run = line.strip()
        elif "[ES STATS] Circuit Breaker" in line:
            match = re.search(r"triggers: (\d+)", line)
            if match:
                cb_triggers = int(match.group(1))
                # Buscamos la siguiente linea para los clones
        elif "[ES STATS] Clone Fallback" in line:
            match = re.search(r"Total Clones Injected: (\d+)", line)
            if match:
                clones = int(match.group(1))
                if current_run:
                    stats.append((current_run, cb_triggers, clones))
                    current_run = None
    
    if not stats:
        print("No se encontraron contadores [ES STATS] en este log.")
        continue
        
    # Agrupar por heuristica
    neural_stats = [s for s in stats if "neural" in s[0]]
    hungarian_stats = [s for s in stats if "hungarian" in s[0]]
    
    def print_summary(name, data):
        if not data: return
        cb_avg = sum(s[1] for s in data) / len(data)
        clones_avg = sum(s[2] for s in data) / len(data)
        print(f"{name} ({len(data)} corridas):")
        print(f"  -> Circuit Breaker Promedio (MAX_FAILURES): {cb_avg:.1f} disparos por corrida")
        print(f"  -> Clones Inyectados Promedio: {clones_avg:.1f} clones por corrida")
        
    print_summary("Neural Surrogate", neural_stats)
    print_summary("Hungarian Exact", hungarian_stats)
    
print("="*80)
