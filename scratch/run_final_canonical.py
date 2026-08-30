import os
import subprocess
import time
import json
import hashlib
import sys
import urllib.request
import pandas as pd

OUTPUT_DIR = "final_canonical_campaign"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 380
SERVER_URL = "http://127.0.0.1:5000"

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SHELLS = [1, 2, 3, 4, 5]
VARIANTS = ["classifier_filter", "full_surrogate"]
CORES = [24] # Solo 24 cores (sin duplicados)

THRESHOLD_MAP = {
    1: 0.70,
    2: 0.70,
    3: 0.60,
    4: 0.70,
    5: 0.70
}

# ─── HASH DE PIPELINE ──────────────────────────────────────────────────────────
PIPELINE_HASH_FILES = [
    "./build/experiment_runner",
    "surrogate_models/surrogate_server.py",
    "surrogate_models/results/regressor_calibration.json",
    "src/neural_heuristic.cpp",
    "surrogate_models/results/surrogate_stats.txt",
]

def compute_pipeline_hash():
    h = hashlib.sha256()
    for path in PIPELINE_HASH_FILES:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    h.update(path.encode())
                    h.update(f.read())
            except Exception: pass
    return h.hexdigest()[:16]

CURRENT_PIPELINE_HASH = compute_pipeline_hash()
# ───────────────────────────────────────────────────────────────────────────────

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

def verify_board_with_astar(board_flat_str, tag="exp1"):
    if not board_flat_str or len(board_flat_str) < 5: return 0, False, False
    rows = [r for r in board_flat_str.split("|") if r.strip() != ""]
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{tag}.sok")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
        
    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): return 0, False, True

    cmd = [solver_bin, temp_file, "0", "500"]
    timed_out = False
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60, text=True)
    except subprocess.TimeoutExpired as e:
        out = e.output if e.output else ""
        if isinstance(out, bytes): out = out.decode('utf-8', errors='replace')
        timed_out = True
    except Exception as e:
        out = getattr(e, "output", "")
        if isinstance(out, bytes): out = out.decode('utf-8', errors='replace')

    pushes = 0
    for line in out.split("\n"):
        if "Pushes:" in line:
            try: pushes = int(line.split(":")[1].strip())
            except: pass
            
    if os.path.exists(temp_file):
        try: os.remove(temp_file)
        except: pass

    if pushes > 0: return pushes, True, False
    elif timed_out: return 0, False, True
    else: return 0, False, False

def run_experiment(heuristic, shell_idx, seed):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_prefix = os.path.join(OUTPUT_DIR, f"{heuristic}_shell{shell_idx}_seed{seed}_cores24")
    out_csv = out_prefix + ".csv"
    out_txt = out_prefix + ".txt"

    runner_path = "./build/experiment_runner"
    
    # Threshold handling
    th = THRESHOLD_MAP.get(shell_idx, 0.70)
    if heuristic != "hungarian":
        resp = set_server_threshold(th)
        print(f"   [API] set_threshold({th}) -> Response: {resp}")
    
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
    
    print(f"🚀 [Canonical] Shell {shell_idx} | Seed {seed:<4} | Var: {heuristic:<18} | Cores 24", flush=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "24"
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
    elapsed = time.time() - start_time
    
    with open(out_txt, "w") as f_out:
        f_out.write(out_text)
        
    return out_txt, out_csv, elapsed

def parse_and_validate(out_txt, out_csv, heuristic, shell_idx, seed, elapsed):
    if not os.path.exists(out_txt) or not os.path.exists(out_csv): return None
    
    df_history = pd.read_csv(out_csv)
    best_boards = []
    
    lines = []
    with open(out_txt, 'r', errors='replace') as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("RANK_"):
            parts = line.split(";")
            if len(parts) >= 4:
                board_str = parts[3].strip()
                n_fit = float(parts[2])
                best_boards.append((board_str, n_fit))
                
    valid_boards = []
    for (bstr, nfit) in best_boards:
        real_p, solved, inconv = verify_board_with_astar(bstr, tag=f"{heuristic}_sh{shell_idx}_sd{seed}")
        valid_boards.append({"pushes": real_p, "solved": solved, "inconclusive": inconv, "neural_fit": nfit})
        
    hybrid_del = 0
    collapsed = 0
    for line in lines:
        if "Hybrid (Hungarian) delegations:" in line:
            try: hybrid_del = int(line.split(":")[-1].strip())
            except: pass
        if "Total false positives filtered:" in line:
            try: collapsed = int(line.split(":")[-1].strip())
            except: pass
            
    best_real_pushes = max([b["pushes"] for b in valid_boards]) if valid_boards else 0
            
    return {
        "Variant": heuristic,
        "Heuristic": heuristic,
        "Shell_Idx": shell_idx,
        "Shell_File": f"shell_{shell_idx}.sok",
        "Seed": seed,
        "Cores": 24,
        "Threshold": THRESHOLD_MAP.get(shell_idx, 0.70),
        "Time_s": elapsed,
        "Hybrid_Delegations_6PlusBoxes": hybrid_del,
        "Collapsed_Immediate": collapsed,
        "Top5_Best_Real_Astar_Pushes": best_real_pushes,
        "pipeline_hash": CURRENT_PIPELINE_HASH
    }

def main():
    print("="*80)
    print(f" 🏁 INICIANDO CAMPAÑA CANÓNICA - 24 CORES, PARÁMETROS ESTRICTOS")
    total_runs = len(VARIANTS) * len(SHELLS) * len(SEEDS)
    print(f" 📊 Total de ejecuciones programadas: {total_runs}")
    print(f" 🔑 Pipeline Hash: {CURRENT_PIPELINE_HASH}")
    print("="*80)
    
    # SANITY CHECK DEL SERVIDOR ANTES DE COMENZAR (Para no fallar 1 hora después)
    print("🔍 [Sanity Check] Verificando conexión con el servidor Flask...")
    set_server_threshold(0.70)
    print("✅ [Sanity Check] ¡Servidor Flask respondiendo correctamente!\n")
    
    results = []
    for heur in VARIANTS:
        for sh in SHELLS:
            for seed in SEEDS:
                txt, csv_f, elap = run_experiment(heur, sh, seed)
                print(f"   ✓ Tiempo: {elap:.2f}s. Evaluando top boards con A*...")
                row = parse_and_validate(txt, csv_f, heur, sh, seed, elap)
                if row:
                    results.append(row)
                    
    df = pd.DataFrame(results)
    final_csv = "final_canonical_campaign_consolidated.csv"
    df.to_csv(final_csv, index=False)
    
    print("\n" + "="*80)
    print(f" 🎉 CAMPAÑA COMPLETADA. CSV Consolidado guardado en {final_csv}")
    print("="*80)

if __name__ == '__main__':
    main()
