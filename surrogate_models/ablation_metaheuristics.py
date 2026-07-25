import subprocess
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for publication
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def run_experiment(algorithm, heuristic, time_limit, out_csv):
    print(f"Running {algorithm} with {heuristic} heuristic for {time_limit} seconds...")
    
    # Base shell for generation
    shell_file = "levels/eval_0.sok" 
    
    # We will use FO1 (Pushes) to see how difficult they can make it
    fitness_type = "FO1"
    seed = "42"
    
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
        subprocess.run(cmd, env=env, timeout=time_limit + 10, check=False, stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"[{algorithm} - {heuristic}] Killed due to timeout.")

def plot_results(algorithms, time_limit):
    print("Generating plots...")
    
    plt.figure(figsize=(12, 6))
    
    colors = {"GA": "#1f77b4", "ES": "#ff7f0e", "SA": "#2ca02c"}
    styles = {"neural": "-", "hungarian": "--"}
    labels = {"neural": "Surrogate (SE-ResNet)", "hungarian": "A* (Fuerza Bruta)"}
    
    data_frames = []
    
    for algo in algorithms:
        for heuristic in ["neural", "hungarian"]:
            csv_file = f"optuna_results/{algo}_{heuristic}_log.csv"
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    # Convert ms to seconds
                    df['time_sec'] = df['time_ms'] / 1000.0
                    
                    df['Algorithm'] = algo
                    df['Heuristic'] = labels[heuristic]
                    data_frames.append(df)
                    
                    # Plot Fitness vs Time
                    plt.plot(df['time_sec'], df['fitness'], 
                             label=f"{algo} + {labels[heuristic]}",
                             color=colors[algo],
                             linestyle=styles[heuristic],
                             linewidth=2)
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

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
    for df in data_frames:
        plt.plot(df['time_sec'], df['evaluations'], 
                 label=f"{df['Algorithm'].iloc[0]} + {df['Heuristic'].iloc[0]}",
                 color=colors[df['Algorithm'].iloc[0]],
                 linestyle=styles["neural"] if "Surrogate" in df['Heuristic'].iloc[0] else styles["hungarian"],
                 linewidth=2)
                 
    plt.title("Nodos Evaluados vs. Tiempo")
    plt.xlabel("Tiempo de Ejecución (segundos)")
    plt.ylabel("Evaluaciones de Fitness (Acumulado)")
    plt.yscale('log') # Log scale is perfect for showing orders of magnitude difference
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"optuna_results/metaheuristics_evals_time.pdf")
    plt.savefig(f"optuna_results/metaheuristics_evals_time.png", dpi=300)
    print("Plots saved in optuna_results/")

if __name__ == "__main__":
    os.makedirs("optuna_results", exist_ok=True)
    
    # We'll use 2 minutes (120s) to keep total experiment time reasonable (12 mins total)
    TIME_LIMIT = 120
    
    # E.g. create a simple 7x7 shell if eval_0.sok doesn't exist
    os.makedirs("levels", exist_ok=True)
    if not os.path.exists("levels/eval_0.sok"):
        with open("levels/eval_0.sok", "w") as f:
            f.write("#######\n#     #\n#     #\n#     #\n#     #\n#     #\n#######")
            
    algorithms = ["ES", "GA", "SA"]
    heuristics = ["neural", "hungarian"]
    
    for algo in algorithms:
        for heuristic in heuristics:
            out_csv = f"optuna_results/{algo}_{heuristic}_log.csv"
            run_experiment(algo, heuristic, TIME_LIMIT, out_csv)
            
    plot_results(algorithms, TIME_LIMIT)
