import os, subprocess, json, time

SHELLS = [1, 2, 3, 4, 5]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
CORES = 24
TIMEOUT = 300

from run_mini_test import verify_board_with_astar

def run():
    results = []
    
    for shell in SHELLS:
        for seed in SEEDS:
            cmd = f"./build/experiment_runner config/shell_{shell}.sok {seed} {CORES} {TIMEOUT}"
            env = os.environ.copy()
            env["USE_SURROGATE"] = "true"
            env["USE_CLASSIFIER_FILTER"] = "false"
            env["CLASSIFIER_THRESHOLD"] = "0.7"
            if shell == 3:
                env["CLASSIFIER_THRESHOLD"] = "0.6"
            
            p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
            
            # Parse Top-5
            top_boards = []
            for line in p.stdout.split('\n') + p.stderr.split('\n'):
                if line.startswith("RANK_"):
                    parts = line.split(";")
                    if len(parts) >= 4:
                        try:
                            n_fit = float(parts[1].strip())
                            s_prob = float(parts[2].strip())
                            b_str = parts[3].strip()
                            if n_fit > -1e8: top_boards.append((s_prob, b_str))
                        except: pass
                        
            for s_prob, b_str in top_boards:
                _, is_sol, is_inc = verify_board_with_astar(b_str)
                if is_sol: res = "SOLVED"
                elif is_inc: res = "INCONCLUSIVE"
                else: res = "DEADLOCK_Genuino"
                
                results.append({"shell": shell, "seed": seed, "prob": s_prob, "result": res})
                print(f"Shell {shell} Seed {seed}: Prob={s_prob:.4f} Result={res}")
                
    with open("scratch/prob_diagnostic_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run()
