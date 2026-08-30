import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "final_paper_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 380
SERVER_URL = "http://127.0.0.1:5000"

THRESHOLD_MAP = {
    1: 0.70,
    2: 0.70,
    3: 0.60,
    4: 0.70,
    5: 0.70
}

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SHELLS = [1, 2, 3, 4, 5]
CORES = [24]

VARIANTS = [
    ("Hybrid Surrogate (Reg)", "hybrid_regressor")
]

def set_server_threshold(threshold, retries=3):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                json.loads(resp.read().decode('utf-8'))
                return True
        except Exception as e:
            time.sleep(2)
    sys.exit(1)

def run_experiment(label, heuristic, shell_idx, seed, cores):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_prefix = os.path.join(OUTPUT_DIR, f"{heuristic}_shell{shell_idx}_seed{seed}")
    out_csv = out_prefix + ".csv"
    out_txt = out_prefix + ".txt"

    runner_path = "./build/experiment_runner"
    
    th = THRESHOLD_MAP.get(shell_idx, 0.70)
    if heuristic != "hungarian":
        set_server_threshold(th)

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--mu", "9",
        "--lambda", "28",
        "--mutRate", "0.8559",
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", out_csv
    ]

    print(f"🚀 Shell {shell_idx} | Seed {seed:<2} | Var: {label} ...", end=" ", flush=True)

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
    elapsed = time.time() - start_time

    with open(out_txt, "w") as f_out:
        f_out.write(out_text)

    # Parse results
    deadlocks_filtered = 0; fp_count = 0; hybrid_del = 0
    top5_accuracy_pct = 0.0; top5_real_pushes = 0.0
    valid_top5 = 0; pushes_sum = 0
    
    for line in out_text.split('\n'):
        if "[ES STATS] Classifier Deadlocks Filtered" in line:
            try: deadlocks_filtered = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Classifier False Positives (Collapses)" in line:
            try: fp_count = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Hybrid Delegations (Hungarian)" in line:
            try: hybrid_del = int(line.split(":")[1].strip())
            except: pass
        elif "RANK_" in line and ";prob;" in line:
            parts = line.split(";")
            # FORMAT: RANK_1;fitness;prob;board_str;real_pushes
            # Wait, the script has 'board' instead of 'prob' if we didn't add real_pushes?
            # Actually, the user's previous script for hybrid_regressor didn't have prob output.
            pass
            
    # For real pushes and accuracy, we can parse the top 5 from the text output:
    # "Solucion A* -> N Pushes" or "Result: DEADLOCK/TIMEOUT"
    # Wait, does the C++ print "Solucion A* -> N pushes" for the final top 5?
    pass

    # Read CSV for top 5
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        if 'pushes_real' in df.columns and 'fitness' in df.columns:
            # top 5
            top5 = df.head(5)
            # definite accuracy = percentage of top 5 that are solved (pushes_real > 0)
            solved = top5[top5['pushes_real'] > 0]
            top5_accuracy_pct = (len(solved) / len(top5)) * 100 if len(top5) > 0 else 0.0
            top5_real_pushes = solved['pushes_real'].mean() if len(solved) > 0 else 0.0
    
    print(f"[{elapsed:.1f}s] Accuracy: {top5_accuracy_pct:.1f}%")

    return {
        "heuristic": heuristic,
        "shell": shell_idx,
        "seed": seed,
        "time_s": elapsed,
        "hybrid_del": hybrid_del,
        "deadlocks": deadlocks_filtered,
        "accuracy": top5_accuracy_pct,
        "pushes": top5_real_pushes
    }

print("\n" + "="*80)
print(" 🚀 INICIANDO CAMPAÑA FINAL DEL PAPER (100 Corridas)")
print("="*80)

results = []
for (lbl, heur) in VARIANTS:
    for sh in SHELLS:
        for s in SEEDS:
            res = run_experiment(lbl, heur, sh, s, 24)
            results.append(res)

# Aggregation
df = pd.DataFrame(results)
print("\n" + "="*80)
print(" 📊 RESULTADOS AGREGADOS POR SHELL Y HEURÍSTICA (Medias sobre 10 semillas)")
print("="*80)

agg = df.groupby(["shell", "heuristic"]).agg({
    "time_s": "mean",
    "pushes": "mean",
    "accuracy": "mean",
    "hybrid_del": "mean",
    "deadlocks": "mean"
}).reset_index()

print(agg.to_string())

# Removed automated speedup ratio computation as baseline is not run.

