import os
import pandas as pd
import numpy as np

results = []
heuristics = ["neural", "hungarian"]
algorithms = ["ES", "GA", "SA"]
shells = [1, 2, 3, 4, 5]

for algo in algorithms:
    for shell_id in shells:
        for heur in heuristics:
            final_fitnesses = []
            final_evals = []
            for seed in range(42, 52):
                csv_path = f"optuna_results/{algo}_{heur}_shell{shell_id}_seed{seed}_log.csv"
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, on_bad_lines='skip')
                        df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
                        df = df.dropna(subset=['time_ms', 'fitness', 'evaluations'])
                        if len(df) > 0:
                            final_fitnesses.append(df['fitness'].iloc[-1])
                            final_evals.append(df['evaluations'].iloc[-1])
                    except Exception:
                        pass
            
            if final_fitnesses:
                results.append({
                    "Algorithm": algo,
                    "Shell": f"Shell {shell_id}",
                    "Heuristic": heur,
                    "Avg Fitness": np.mean(final_fitnesses),
                    "Std Fitness": np.std(final_fitnesses),
                    "Avg Evals": np.mean(final_evals),
                    "Runs": len(final_fitnesses)
                })

df_results = pd.DataFrame(results)

print("="*80)
print("FINAL BENCHMARK SUMMARY")
print("="*80)
print(df_results.to_markdown(index=False))
print("="*80)

# Compare Neural vs Hungarian
print("\nNEURAL VS HUNGARIAN COMPARISON (HIGHER FITNESS IS BETTER)")
print("-" * 60)
for algo in algorithms:
    for shell_id in shells:
        shell_name = f"Shell {shell_id}"
        subset = df_results[(df_results['Algorithm'] == algo) & (df_results['Shell'] == shell_name)]
        if len(subset) == 2:
            neural = subset[subset['Heuristic'] == 'neural'].iloc[0]
            hungarian = subset[subset['Heuristic'] == 'hungarian'].iloc[0]
            
            winner = "NEURAL" if neural['Avg Fitness'] > hungarian['Avg Fitness'] else "HUNGARIAN"
            if abs(neural['Avg Fitness'] - hungarian['Avg Fitness']) < 1e-5:
                winner = "TIE"
                
            print(f"{algo} | {shell_name} -> Winner: {winner}")
            print(f"   Neural:    {neural['Avg Fitness']:.2f} ± {neural['Std Fitness']:.2f} (Evals: {neural['Avg Evals']:.0f})")
            print(f"   Hungarian: {hungarian['Avg Fitness']:.2f} ± {hungarian['Std Fitness']:.2f} (Evals: {hungarian['Avg Evals']:.0f})")

