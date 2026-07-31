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

def run_experiment(algorithm, heuristic, time_limit, seed, shell_file, out_csv):
    print(f"Running {algorithm} with {heuristic} heuristic on {shell_file} for {time_limit}s (Seed {seed})...")
    
    # Assert that the shell file exists
    assert os.path.exists(shell_file), f"Shell file {shell_file} not found!"
            
    # We will use FO1 (Pushes) to see how difficult they can make it
    fitness_type = "FO1"
    
    tmp_csv = out_csv + ".tmp"
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)
    
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
        "--maxEvals", "1000000000",
        "--out_csv", tmp_csv
    ]
    
    try:
        # Give a small grace period (10s) over the time limit for graceful shutdown
        # Capture stderr to detect crashes
        # Capture stderr to detect crashes, and stdout to get diversity metrics
        result = subprocess.run(cmd, env=env, timeout=time_limit + 120, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        for line in result.stdout.split("\n"):
            if "[DIVERSITY]" in line or "[ES STATS]" in line:
                print(f"  {line.strip()}")
                
        if result.returncode != 0:
            print(f"[{algorithm} - {heuristic}] Exit code {result.returncode}. Stderr:\n{result.stderr}")
        else:
            if os.path.exists(tmp_csv):
                os.rename(tmp_csv, out_csv)
    except subprocess.TimeoutExpired:
        print(f"[{algorithm} - {heuristic}] Killed due to timeout.")
        if os.path.exists(tmp_csv):
            os.rename(tmp_csv, out_csv)
    except Exception as e:
        print(f"[{algorithm} - {heuristic}] Crash/Error: {e}")

def plot_results(algorithms, heuristics, seeds, shells, time_limit):
    print("Generating plots per shell...")
    
    colors = {"GA": "#1f77b4", "ES": "#ff7f0e", "SA": "#2ca02c"}
    styles = {"neural": "-", "hungarian": "--"}
    labels = {"neural": "Surrogate (SE-ResNet, GPU)", "hungarian": "A* Exacto (1-CPU)"}
    
    # Create common time grid for interpolation
    time_grid = np.linspace(0, time_limit, 200)
    
    for shell_id, shell in enumerate(shells, 1):
        agg_fitness = {}
        agg_evals = {}
        
        for algo in algorithms:
            for heuristic in heuristics:
                key = f"{algo}_{heuristic}"
                agg_fitness[key] = []
                agg_evals[key] = []
                
                for seed in seeds:
                    csv_file = f"optuna_results/{algo}_{heuristic}_shell{shell_id}_seed{seed}_log.csv"
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
        
        # 1st Plot: Fitness vs Time for this shell
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
                    
        plt.title(f"Evolución del Fitness vs. Tiempo (Shell {shell_id})")
        plt.xlabel("Tiempo de Ejecución (segundos)")
        plt.ylabel("Dificultad Alcanzada (Fitness: Empujes)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"optuna_results/metaheuristics_ablation_time_shell{shell_id}.pdf")
        plt.close()
        
        # 2nd Plot: Evaluations vs Time for this shell
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
                     
        plt.title(f"Nodos Evaluados vs. Tiempo (Shell {shell_id})")
        plt.xlabel("Tiempo de Ejecución (segundos)")
        plt.ylabel("Evaluaciones de Fitness (Acumulado)")
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"optuna_results/metaheuristics_evals_time_shell{shell_id}.pdf")
        plt.close()
        
    print("Plots saved in optuna_results/")

def test_determinism():
    print("Running determinism test for GA + neural (seed 42)...")
    shell_file = "levels/shell_1.sok"
    csv1 = "optuna_results/determinism_1.csv"
    csv2 = "optuna_results/determinism_2.csv"
    
    run_experiment("GA", "neural", 10, "42", shell_file, csv1)
    run_experiment("GA", "neural", 10, "42", shell_file, csv2)
    
    df1 = pd.read_csv(csv1, on_bad_lines='skip')
    df2 = pd.read_csv(csv2, on_bad_lines='skip')
    
    fit1 = df1['fitness'].iloc[-1]
    fit2 = df2['fitness'].iloc[-1]
    evals1 = df1['evaluations'].iloc[-1]
    evals2 = df2['evaluations'].iloc[-1]
    
    print(f"Run 1 -> Fitness: {fit1}, Evals: {evals1}")
    print(f"Run 2 -> Fitness: {fit2}, Evals: {evals2}")
    
    if fit1 == fit2 and evals1 == evals2:
        print("Determinism test PASSED.")
    else:
        print("Determinism test FAILED.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ALL", help="Algorithm to run")
    args = parser.parse_args()

    os.makedirs("optuna_results", exist_ok=True)
    
    # Test determinism first (only on PC 1, or if ALL)
    if args.algo == "ALL" or args.algo == "GA":
        test_determinism()
    
    TIME_LIMIT = 120
    
    if args.algo == "ALL":
        algorithms = ["ES", "GA", "SA"]
    else:
        algorithms = [args.algo]
        
    time_limit = TIME_LIMIT
    heuristics = ["neural", "hungarian"]
    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    seeds = [str(i) for i in range(42, 52)]
    
    # 3. Main Experiment Loop
    print("\nStarting Main Benchmark Loop...")
    for shell_file in shells:
        for algo in algorithms:
            for heuristic in heuristics:
                for seed in seeds:
                    shell_id = shell_file.split("_")[-1].split(".")[0]
                    out_csv = f"optuna_results/{algo}_{heuristic}_shell{shell_id}_seed{seed}_log.csv"
                    if os.path.exists(out_csv):
                        print(f"Skipping {algo} + {heuristic} on {shell_file} (Seed {seed}) - Already done.")
                        continue
                        
                    run_experiment(algo, heuristic, time_limit, seed, shell_file, out_csv)
            
    plot_results(algorithms, heuristics, seeds, shells, time_limit)
