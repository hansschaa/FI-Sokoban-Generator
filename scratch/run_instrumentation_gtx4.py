import os
import subprocess
import sys

def main():
    print("--- 1. Compilando experiment_runner (con instrumentacion ya inyectada) ---")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..") # Ensure root

    subprocess.run(["make", "-C", "build", "-j12", "experiment_runner"], check=True, stdout=subprocess.DEVNULL)

    print("\n--- 2. Ejecutando Semilla 44 (Full Surrogate) en Shell 5 ---")
    cmd = [
        "./build/experiment_runner",
        "GA",
        "FO6",
        "44",
        "tuning/Instances/shell_577.txt",
        "--heuristic", "full_surrogate",
        "--maxEvals", "50",
        "--timeLimit", "100"
    ]
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "24"
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    print("\n=== LOG CRUDO DE INSTRUMENTACION (SEMILLA 44) ===")
    for line in res.stderr.split('\n'):
        if "[TIMING_PHASE]" in line or "Error:" in line or "Warning:" in line or "[TIMING_INIT]" in line:
            print(line)
            
    print("\n=================================================")
    print("Por favor enviame el texto completo entre las lineas de '='.")

if __name__ == "__main__":
    main()
