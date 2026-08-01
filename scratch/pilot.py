import os
import pandas as pd
import numpy as np

results_dir = "optuna_results/"
algorithms = ["ES", "GA", "SA"]
heuristics = ["neural", "hungarian"]
seeds = [str(i) for i in range(42, 52)]
shells = ["shell1", "shell2", "shell3", "shell4", "shell5"]

def main():
    print("Pilot Analysis of Partial Results\n")
    
    for shell in shells:
        shell_data = {}
        for algo in algorithms:
            for heuristic in heuristics:
                key = f"{algo}_{heuristic}"
                fitness_vals = []
                evals_vals = []
                for seed in seeds:
                    fname = f"{algo}_{heuristic}_{shell}_seed{seed}_log.csv"
                    fpath = os.path.join(results_dir, fname)
                    if os.path.exists(fpath):
                        try:
                            df = pd.read_csv(fpath, on_bad_lines='skip')
                            if len(df) >= 1:
                                fitness_vals.append(df['fitness'].iloc[-1])
                                evals_vals.append(df['evaluations'].iloc[-1])
                        except Exception:
                            pass
                
                if fitness_vals:
                    shell_data[key] = {
                        'fitness': fitness_vals,
                        'evals': evals_vals,
                        'count': len(fitness_vals)
                    }
        
        if not shell_data:
            continue
            
        print(f"=== {shell.upper()} ===")
        for algo in algorithms:
            neural_key = f"{algo}_neural"
            hungarian_key = f"{algo}_hungarian"
            
            if neural_key in shell_data or hungarian_key in shell_data:
                print(f"Algorithm: {algo}")
                
                if neural_key in shell_data:
                    fit_mean = np.mean(shell_data[neural_key]['fitness'])
                    fit_std = np.std(shell_data[neural_key]['fitness'])
                    evals_mean = np.mean(shell_data[neural_key]['evals'])
                    count = shell_data[neural_key]['count']
                    print(f"  Neural Surrogate : Fitness = {fit_mean:.2f} +- {fit_std:.2f} | Evals = {evals_mean:.0f} (N={count})")
                    
                if hungarian_key in shell_data:
                    fit_mean = np.mean(shell_data[hungarian_key]['fitness'])
                    fit_std = np.std(shell_data[hungarian_key]['fitness'])
                    evals_mean = np.mean(shell_data[hungarian_key]['evals'])
                    count = shell_data[hungarian_key]['count']
                    print(f"  Hungarian Exact  : Fitness = {fit_mean:.2f} +- {fit_std:.2f} | Evals = {evals_mean:.0f} (N={count})")
                
                if neural_key in shell_data and hungarian_key in shell_data:
                    wins, ties, valid = 0, 0, 0
                    for i in range(len(seeds)):
                        if i < len(shell_data[neural_key]['fitness']) and i < len(shell_data[hungarian_key]['fitness']):
                            valid += 1
                            fn = shell_data[neural_key]['fitness'][i]
                            fh = shell_data[hungarian_key]['fitness'][i]
                            if fn > fh: wins += 1
                            elif fn == fh: ties += 1
                    print(f"  Neural vs Hungarian: {wins} Wins, {ties} Ties, {valid - wins - ties} Losses (out of {valid} common seeds)\n")
                else:
                    print()

if __name__ == "__main__":
    main()
