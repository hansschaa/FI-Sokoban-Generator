import os
import subprocess
import time
import json
import hashlib
import sys
import urllib.request
import pandas as pd

OUTPUT_DIR = "final_parallel_campaign"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 380
SERVER_URL = "http://127.0.0.1:5000"

SEEDS = list(range(42, 52))
SHELLS = [1, 2, 3, 4, 5]

# Vamos a correr TODAS las variantes para tener la tabla completa y limpia bajo el mismo budget histórico
VARIANTS = [
    "hungarian", 
    "classifier_filter", 
    "hybrid_regressor", 
    "full_surrogate",
    "full_surrogate_no_audit"
]

CORES = [24] # GTX4 cores

THRESHOLD_MAP = {
    1: 0.70,
    2: 0.70,
    3: 0.60,
    4: 0.70,
    5: 0.70
}

def set_server_threshold(threshold, retries=3):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp_data = json.loads(resp.read().decode('utf-8'))
                return resp_data
        except Exception as e:
            time.sleep(2)
    print(f"❌ CRÍTICO: No se pudo confirmar umbral={threshold} en el server tras {retries} intentos. Abortando.")
    sys.exit(1)

def run_experiment(heuristic, shell_idx, seed):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_prefix = os.path.join(OUTPUT_DIR, f"{heuristic}_shell{shell_idx}_seed{seed}_cores24")
    out_csv = out_prefix + ".csv"
    out_txt = out_prefix + ".txt"

    runner_path = "./build/experiment_runner"
    
    # Threshold handling (solo para los que usan neural)
    th = THRESHOLD_MAP.get(shell_idx, 0.70)
    if heuristic not in ["hungarian", "manhattan", "simple"]:
        resp = set_server_threshold(th)
        print(f"   [API] set_threshold({th}) -> Response: {resp}")
    
    # MUY IMPORTANTE: BUDGET HISTÓRICO (OPCIÓN B de Claude)
    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", out_csv,
        "--mu", "9",
        "--lambda", "28",
        "--mutRate", "0.8559",
        "--stagLimit", "199"
    ]
    
    print(f"🚀 [Final Campaign] Shell {shell_idx} | Seed {seed:<4} | Var: {heuristic:<18}", flush=True)

    env = os.environ.copy()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
    except Exception as e:
        out_text = getattr(e, 'stdout', str(e))
    elapsed = time.time() - start_time
    
    with open(out_txt, "w") as f_out:
        f_out.write(out_text)
        
    return out_txt, out_csv, elapsed

def main():
    print("=== INICIANDO CAMPAÑA FINAL (PARALELIZADA) ===")
    print("Budget Histórico (Opción B): mu=9, lambda=28, mutRate=0.8559, stagLimit=199")
    print(f"Variantes a correr: {VARIANTS}")
    
    total_tasks = len(VARIANTS) * len(SHELLS) * len(SEEDS)
    current = 0
    
    for shell_idx in SHELLS:
        for seed in SEEDS:
            for variant in VARIANTS:
                current += 1
                print(f"\n--- Tarea {current}/{total_tasks} ---")
                run_experiment(variant, shell_idx, seed)
                
    print("\n✅ ¡CAMPAÑA FINAL COMPLETADA!")
    print(f"Los resultados y logs están guardados en el directorio: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
