import subprocess
import time
import re
import sys

def run_seed(seed):
    cmd = [
        "./build/experiment_runner", "ES", "FO1", str(seed), "levels/shell_1.sok",
        "--heuristic", "full_surrogate",
        "--timeLimit", "300",
        "--maxEvals", "1000000",
        "--mu", "9",
        "--lambda", "28",
        "--mutRate", "0.8559",
        "--stagLimit", "199"
    ]
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    
    lines = res.stdout.split('\n')
    pushes = 0
    for line in reversed(lines):
        if ';' in line:
            parts = line.split(';')
            if len(parts) >= 3:
                pushes = int(float(parts[0]) * -1)
                break
                
    return end - start, pushes

def main():
    print("=== PRUEBA DE CACHÉ DE SERVIDOR ===")
    print("Corriendo Seed 42 por primera vez (debería ser lento si el caché está frío)...")
    t1, p1 = run_seed(42)
    print(f"Corrida 1: {t1:.2f}s | Pushes: {p1}")
    
    print("\nCorriendo Seed 42 inmediatamente después (debería ser rápido si hay caché)...")
    t2, p2 = run_seed(42)
    print(f"Corrida 2: {t2:.2f}s | Pushes: {p2}")

if __name__ == "__main__":
    main()
