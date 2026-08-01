import os
import subprocess
import time
import sys
import concurrent.futures

def run_single(seed, shell):
    runner_path = "./build/experiment_runner"
    cmd = [
        runner_path, "ES", "FO1", seed, shell,
        "--heuristic", "neural",
        "--timeLimit", "300",
        "--maxEvals", "1000000"
    ]
    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=310, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes):
            out_text = out_text.decode('utf-8', errors='replace')
    elapsed = time.time() - start_time
            
    disyuntor_count = -1
    delegations = -1
    gens = 0
    evals = 0
    for line in out_text.split('\n'):
        if "[ES STATS] Circuit Breaker (MAX_FAILURES) triggers:" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    disyuntor_count = int(parts[1].strip())
                except:
                    pass
        elif "[ES STATS] Hybrid Hungarian Delegations (box_count >= 6):" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    delegations = int(parts[1].strip())
                except:
                    pass
        elif "[ES STATS] Total Generations:" in line:
            parts = line.split("|")
            for p in parts:
                if "Generations:" in p:
                    try:
                        gens = int(p.split(":")[1].strip())
                    except:
                        pass
                elif "Evals:" in p:
                    try:
                        evals = int(p.split(":")[1].strip())
                    except:
                        pass
    return (seed, shell, disyuntor_count, delegations, gens, evals, elapsed)

def main():
    seeds = [str(i) for i in range(42, 52)]
    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    
    tasks = [(s, sh) for sh in shells for s in seeds]
    results = []
    
    print("Starting 50 runs...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_single, s, sh) for s, sh in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"{res[1]} Seed {res[0]}: Triggers={res[2]}, Delegations={res[3]}, Gens={res[4]}, Time={res[6]:.1f}s", flush=True)
            
    total_triggers = 0
    valid_runs = 0
    
    shell_data = {}
    for s, sh, count, deleg, gens, evals, elapsed in results:
        if count != -1:
            total_triggers += count
            valid_runs += 1
            shell_data.setdefault(sh, []).append((count, deleg, gens, evals, elapsed))

    if valid_runs > 0:
        print(f"\n=== FINAL SUMMARY (across {valid_runs} runs) ===", flush=True)
        print(f"Global Avg Disyuntor triggers per run: {total_triggers / valid_runs:.2f}", flush=True)
        for sh in sorted(shell_data.keys()):
            runs = shell_data[sh]
            avg_trig = sum(r[0] for r in runs) / len(runs)
            avg_deleg = sum(r[1] for r in runs if r[1] != -1) / len(runs) if runs else 0
            avg_gens = sum(r[2] for r in runs) / len(runs)
            avg_evals = sum(r[3] for r in runs) / len(runs)
            avg_time = sum(r[4] for r in runs) / len(runs)
            tput = (avg_evals / avg_time) if avg_time > 0 else 0
            print(f"\n{sh} (from {len(runs)} runs):", flush=True)
            print(f"  Avg Disyuntor triggers: {avg_trig:.2f}", flush=True)
            print(f"  Avg Hybrid Delegations (box>=6 to Hungarian): {avg_deleg:.2f}", flush=True)
            print(f"  Avg Generations: {avg_gens:.1f} | Avg Evals: {avg_evals:.1f} | Avg Time: {avg_time:.2f}s | Throughput: {tput:.1f} evals/sec", flush=True)
            
if __name__ == '__main__':
    main()
