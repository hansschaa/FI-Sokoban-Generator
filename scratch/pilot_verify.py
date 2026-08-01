import os
import subprocess
import time
import pandas as pd

def run_pilot():
    print("=== STARTING PILOT VERIFICATION ===")
    
    seeds = ["90", "91", "92"]
    shells = [f"levels/shell_{i}.sok" for i in range(1, 3)]
    algorithms = ["ES"]
    heuristics = ["neural", "hungarian"]
    
    os.makedirs("optuna_results", exist_ok=True)
    
    for shell in shells:
        shell_id = shell.split('_')[1].split('.')[0]
        for algo in algorithms:
            for h in heuristics:
                for s in seeds:
                    fname = f"optuna_results/{algo}_{h}_shell{shell_id}_seed{s}_log.csv"
                    if os.path.exists(fname): os.remove(fname)
                    if os.path.exists(fname + ".tmp"): os.remove(fname + ".tmp")
                        
    TIME_LIMIT = 15
    
    import signal
    print("Starting Surrogate Server for Pilot...")
    server_process = subprocess.Popen(["./venv/bin/python3", "surrogate_models/surrogate_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = server_process.stdout.readline()
        if not line: break
        if "Server ready" in line: break

    try:
        for shell in shells:
            shell_id = shell.split('_')[1].split('.')[0]
            for algo in algorithms:
                for heuristic in heuristics:
                    for seed in seeds:
                        print(f"\nRunning {algo} with {heuristic} heuristic on {shell} for {TIME_LIMIT}s (Seed {seed})...")
                    
                    out_csv = f"optuna_results/{algo}_{heuristic}_shell{shell_id}_seed{seed}_log.csv"
                    tmp_csv = out_csv + ".tmp"
                    
                    cmd = [
                        "./build/experiment_runner",
                        algo,
                        "FO1",
                        seed,
                        shell,
                        "--heuristic", heuristic,
                        "--timeLimit", str(TIME_LIMIT),
                        "--maxEvals", "1000000",
                        "--out_csv", tmp_csv
                    ]
                    
                    env = os.environ.copy()
                    
                    try:
                        result = subprocess.run(cmd, env=env, timeout=TIME_LIMIT + 120, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        if result.returncode != 0:
                            print(f"[{algo} - {heuristic}] Exit code {result.returncode}.")
                            print("\n".join(result.stdout.strip().split("\n")[-10:]))
                        else:
                            if os.path.exists(tmp_csv): os.rename(tmp_csv, out_csv)
                            
                        for line in result.stdout.split("\n"):
                            if "[DIVERSITY]" in line:
                                print(f"  {line.strip()}")
                                
                    except subprocess.TimeoutExpired:
                        print(f"[{algo} - {heuristic}] Killed due to timeout.")
                        if os.path.exists(tmp_csv):
                            os.rename(tmp_csv, out_csv)
                        else:
                            print("  WARNING: tmp_csv does not exist. The process hung before writing anything.")
    finally:
        print("Killing server...")
        server_process.send_signal(signal.SIGINT)
        server_process.wait()

    print("\n=== PILOT VERIFICATION RESULTS ===")
    for shell in shells:
        shell_id = shell.split('_')[1].split('.')[0]
        print(f"\nShell {shell_id}:")
        for algo in algorithms:
            for heuristic in heuristics:
                completed = 0
                total_evals = 0
                for seed in seeds:
                    fname = f"optuna_results/{algo}_{heuristic}_shell{shell_id}_seed{seed}_log.csv"
                    if os.path.exists(fname):
                        df = pd.read_csv(fname, on_bad_lines='skip')
                        if len(df) >= 1:
                            completed += 1
                            total_evals += df['evaluations'].iloc[-1]
                print(f"  {algo} - {heuristic}: N = {completed}/{len(seeds)} | Avg Evals = {total_evals/max(completed, 1):.0f}")

if __name__ == "__main__":
    run_pilot()
