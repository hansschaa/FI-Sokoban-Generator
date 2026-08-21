import subprocess
import time

shells = ["levels/shell_1.sok"]
seeds = [43, 44, 45, 46, 47]
runner = "./build/experiment_runner"

for seed in seeds:
    cmd = [runner, "ES", "FO1", str(seed), shells[0], "--heuristic", "full_surrogate", "--timeLimit", "300", "--maxEvals", "1000000"]
    start = time.time()
    try:
        # Timeout en 35 segundos para asegurarnos de que pasó la etapa de riesgo de estancamiento temprano (que cae en ~2s)
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        elapsed = time.time() - start
        
        if elapsed < 5:
            print(f"Seed {seed}: COLLAPSE (Time: {elapsed:.2f}s) - Cayó en el mínimo local inmediatamente.")
        else:
            print(f"Seed {seed}: SURVIVED (Time: {elapsed:.2f}s) - Logró evadir el colapso temprano.")
            
    except subprocess.TimeoutExpired:
        print(f"Seed {seed}: SURVIVED (Timeout 35s reached) - Exploración activa y saludable.")
