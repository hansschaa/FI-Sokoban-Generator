import os
import subprocess
import time
import sys
import concurrent.futures
import threading

def run_single(seed, shell):
    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path):
        runner_path = "./build2/experiment_runner"
    
    cmd = [
        runner_path, "ES", "FO1", seed, shell,
        "--heuristic", "neural",
        "--timeLimit", "300",
        "--maxEvals", "1000000",
        "--out_csv", f"scratch/temp_pilot_{seed}_{os.path.basename(shell)}.csv"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=310, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes):
            out_text = out_text.decode('utf-8', errors='replace')
            
    disyuntor_count = "TIMEOUT"
    for line in out_text.split('\n'):
        if "[ES STATS] Circuit Breaker (MAX_FAILURES) triggers:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    disyuntor_count = int(parts[1].strip())
                except:
                    pass
    return disyuntor_count

def run_pilot():
    seeds = [str(i) for i in range(42, 52)]
    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    
    print("Starting Surrogate Server...")
    server_process = subprocess.Popen(
        [sys.executable, "surrogate_models/surrogate_server.py"], 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    def drain():
        for line in server_process.stdout:
            pass
    threading.Thread(target=drain, daemon=True).start()
    
    time.sleep(10) # wait for server to start
    
    total_disyuntor_triggers = 0
    total_runs = 0
    
    tasks = []
    for shell in shells:
        for seed in seeds:
            tasks.append((seed, shell))
            
    print(f"Running {len(tasks)} tasks in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_single, seed, shell): (seed, shell) for seed, shell in tasks}
        for future in concurrent.futures.as_completed(futures):
            seed, shell = futures[future]
            try:
                disyuntor_count = future.result()
                if isinstance(disyuntor_count, int):
                    total_disyuntor_triggers += disyuntor_count
                print(f"[{shell} | Seed {seed}] Disyuntor triggers: {disyuntor_count}")
                total_runs += 1
            except Exception as exc:
                print(f"[{shell} | Seed {seed}] Generated an exception: {exc}")

    print("\n================ PILOT RESULTS ================")
    if total_runs > 0:
        avg = total_disyuntor_triggers / total_runs
        print(f"Total runs completed: {total_runs}/50")
        print(f"Average Disyuntor triggers per run: {avg:.2f}")
    else:
        print("No runs completed successfully.")
        
    print("Killing server...")
    server_process.kill()

if __name__ == '__main__':
    run_pilot()
