import subprocess
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for publication
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def run_experiment(algorithm, heuristic, time_limit, seed, out_csv):
    print(f"Running {algorithm} with {heuristic} heuristic for {time_limit} seconds (Seed {seed})...")
    
    # Base shell for generation
    shell_file = "levels/eval_0.sok"
    
    # Assert that the shell file exists and is not the trivial placeholder
    assert os.path.exists(shell_file), f"Shell file {shell_file} not found!"
    with open(shell_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 5 and not all(c in '# \n' for l in lines for c in l), \
            "eval_0.sok appears to be the trivial empty placeholder. Please provide a real Sokoban shell level."
            
    # We will use FO1 (Pushes) to see how difficult they can make it
    fitness_type = "FO1"
    
    # Ensure the csv is deleted before starting
    if os.path.exists(out_csv):
        os.remove(out_csv)
    
    # Avoid PyTorch thread explosion
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    
    # Executable for main_experiment
    cmd = [
        "./build/experiment_runner", 
        algorithm,
        fitness_type,
        seed,
        shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(time_limit),
        "--out_csv", out_csv
    ]
    
    try:
        # Give a small grace period (10s) over the time limit for graceful shutdown
        # Capture stderr to detect crashes
        result = subprocess.run(cmd, env=env, timeout=time_limit + 10, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"[{algorithm} - {heuristic}] Exit code {result.returncode}. Stderr:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"[{algorithm} - {heuristic}] Killed due to timeout.")
    except Exception as e:
        print(f"[{algorithm} - {heuristic}] Crash/Error: {e}")

def plot_results(algorithms, heuristics, seeds, time_limit):
    print("Generating plots...")
    
    colors = {"GA": "#1f77b4", "ES": "#ff7f0e", "SA": "#2ca02c"}
    styles = {"neural": "-", "hungarian": "--"}
    labels = {"neural": "Surrogate (SE-ResNet, GPU)", "hungarian": "A* (Fuerza Bruta, 1-CPU)"}
    
    # Create common time grid for interpolation
    time_grid = np.linspace(0, time_limit, 200)
    
    # Dictionaries to store interpolated histories
    agg_fitness = {}
    agg_evals = {}
    
    for algo in algorithms:
        for heuristic in heuristics:
            key = f"{algo}_{heuristic}"
            agg_fitness[key] = []
            agg_evals[key] = []
            
            for seed in seeds:
                csv_file = f"optuna_results/{algo}_{heuristic}_seed{seed}_log.csv"
                if os.path.exists(csv_file):
                    try:
                        # on_bad_lines handles potentially truncated csvs due to timeout
                        df = pd.read_csv(csv_file, on_bad_lines='skip')
                        if len(df) < 2:
                            continue
                        
                        df['time_sec'] = df['time_ms'] / 1000.0
                        
                        # Interpolate to common time grid
                        interp_fit = np.interp(time_grid, df['time_sec'], df['fitness'])
                        interp_eval = np.interp(time_grid, df['time_sec'], df['evaluations'])
                        
                        agg_fitness[key].append(interp_fit)
                        agg_evals[key].append(interp_eval)
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
    
    # 1st Plot: Fitness vs Time
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        for heuristic in heuristics:
            key = f"{algo}_{heuristic}"
            if len(agg_fitness[key]) > 0:
                mean_fit = np.mean(agg_fitness[key], axis=0)
                std_fit = np.std(agg_fitness[key], axis=0)
                
                plt.plot(time_grid, mean_fit, label=f"{algo} + {labels[heuristic]}",
                         color=colors[algo], linestyle=styles[heuristic], linewidth=2)
                plt.fill_between(time_grid, mean_fit - std_fit, mean_fit + std_fit,
                                 color=colors[algo], alpha=0.15)
                
    plt.title("Evolución del Fitness vs. Tiempo (Surrogate vs A*)")
    plt.xlabel("Tiempo de Ejecución (segundos)")
    plt.ylabel("Dificultad Alcanzada (Fitness: Empujes)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"optuna_results/metaheuristics_ablation_time.pdf")
    plt.savefig(f"optuna_results/metaheuristics_ablation_time.png", dpi=300)
    plt.close()
    
    # 2nd Plot: Evaluations vs Time
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        for heuristic in heuristics:
            key = f"{algo}_{heuristic}"
            if len(agg_evals[key]) > 0:
                mean_evals = np.mean(agg_evals[key], axis=0)
                std_evals = np.std(agg_evals[key], axis=0)
                
                # Para escala logaritmica, evitar valores <= 0 en limite inferior
                lower_bound = np.maximum(mean_evals - std_evals, 1)
                upper_bound = mean_evals + std_evals
                
                plt.plot(time_grid, mean_evals, label=f"{algo} + {labels[heuristic]}",
                         color=colors[algo], linestyle=styles[heuristic], linewidth=2)
                plt.fill_between(time_grid, lower_bound, upper_bound,
                                 color=colors[algo], alpha=0.15)
                 
    plt.title("Nodos Evaluados vs. Tiempo")
    plt.xlabel("Tiempo de Ejecución (segundos)")
    plt.ylabel("Evaluaciones de Fitness (Acumulado)")
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"optuna_results/metaheuristics_evals_time.pdf")
    plt.savefig(f"optuna_results/metaheuristics_evals_time.png", dpi=300)
    plt.close()
    
    print("Plots saved in optuna_results/")

if __name__ == "__main__":
    os.makedirs("optuna_results", exist_ok=True)
    
    # Limit to 120s per run
    TIME_LIMIT = 120
    
    os.makedirs("levels", exist_ok=True)
    
    algorithms = ["ES", "GA", "SA"]
    heuristics = ["neural", "hungarian"]
    seeds = ["42", "43", "44", "45", "46"]
    
    for algo in algorithms:
        for heuristic in heuristics:
            for seed in seeds:
                out_csv = f"optuna_results/{algo}_{heuristic}_seed{seed}_log.csv"
                # Check if it was already run (useful for resuming if it crashes)
                if not os.path.exists(out_csv):
                    run_experiment(algo, heuristic, TIME_LIMIT, seed, out_csv)
                else:
                    print(f"Skipping {algo} {heuristic} seed {seed}, CSV exists.")
            
    plot_results(algorithms, heuristics, seeds, TIME_LIMIT)
