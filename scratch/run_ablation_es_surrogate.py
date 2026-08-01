import os
import subprocess
import time
import sys
import concurrent.futures
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual para publicación
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

OUTPUT_DIR = "ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300  # 5 minutos de tiempo de circuito por corrida
PYTHON_TIMEOUT = 480  # Protección de 8 minutos para permitir cierre elegante del C++

def run_single_experiment(algo, heuristic, seed, shell_idx):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = os.path.join(OUTPUT_DIR, f"{algo}_{heuristic}_shell{shell_idx}_seed{seed}_log.csv")
    tmp_csv = out_csv + ".tmp"
    if os.path.exists(tmp_csv):
        try:
            os.remove(tmp_csv)
        except:
            pass
            
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        # Ya completado en ejecución anterior
        return (heuristic, seed, shell_idx, True, "Already executed", 0, 0, 0, 0, 0.0)

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path):
        runner_path = "./build2/experiment_runner"

    cmd = [
        runner_path, algo, "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes):
            out_text = out_text.decode('utf-8', errors='replace')
    elapsed = time.time() - start_time

    if os.path.exists(tmp_csv):
        os.rename(tmp_csv, out_csv)

    disyuntor_count = 0
    delegations = 0
    gens = 0
    evals = 0
    best_fitness = -1e9

    for line in out_text.split('\n'):
        if "[ES STATS] Circuit Breaker (MAX_FAILURES) triggers:" in line:
            try: disyuntor_count = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Hybrid Hungarian Delegations (box_count >= 6):" in line:
            try: delegations = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Total Generations:" in line:
            parts = line.split("|")
            for p in parts:
                if "Generations:" in p:
                    try: gens = int(p.split(":")[1].strip())
                    except: pass
                elif "Evals:" in p:
                    try: evals = int(p.split(":")[1].strip())
                    except: pass
        elif line.strip() and ";" in line.strip():
            # Parsear línea final de experimento: -best.fitness;hash;string
            parts = line.strip().split(";")
            if len(parts) >= 3:
                try:
                    best_fitness = -float(parts[0].strip())
                except:
                    pass

    # Si no se pudo parsear de stdout, intentar extraer de la última línea del CSV
    if best_fitness == -1e9 and os.path.exists(out_csv):
        try:
            df = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df) > 0 and 'fitness' in df.columns:
                best_fitness = float(df['fitness'].iloc[-1])
                if evals == 0 and 'evaluations' in df.columns:
                    evals = int(df['evaluations'].iloc[-1])
        except:
            pass

    return (heuristic, seed, shell_idx, False, out_text, disyuntor_count, delegations, gens, evals, best_fitness, elapsed)

def analyze_and_plot(seeds, shells):
    print("\n" + "="*80)
    print(" 📊 ANÁLISIS DE ABLACIÓN PRE-REGISTRADO Y GENERACIÓN DE GRÁFICOS")
    print("="*80)

    summary_data = []
    
    # Evaluar por combinación Shell
    wins = 0
    ties = 0
    losses = 0
    total_shells = len(shells)

    for shell_idx in shells:
        shell_fit_hungarian = []
        shell_fit_neural = []
        shell_eval_hungarian = []
        shell_eval_neural = []
        
        for seed in seeds:
            for heur in ["hungarian", "neural"]:
                csv_path = os.path.join(OUTPUT_DIR, f"ES_{heur}_shell{shell_idx}_seed{seed}_log.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, on_bad_lines='skip')
                        if len(df) > 0:
                            final_fit = float(df['fitness'].iloc[-1])
                            final_evals = float(df['evaluations'].iloc[-1])
                            if heur == "hungarian":
                                shell_fit_hungarian.append(final_fit)
                                shell_eval_hungarian.append(final_evals)
                            else:
                                shell_fit_neural.append(final_fit)
                                shell_eval_neural.append(final_evals)
                    except Exception as e:
                        pass
        
        mean_fit_hung = np.mean(shell_fit_hungarian) if shell_fit_hungarian else 0.0
        mean_fit_neur = np.mean(shell_fit_neural) if shell_fit_neural else 0.0
        mean_ev_hung  = np.mean(shell_eval_hungarian) if shell_eval_hungarian else 0.0
        mean_ev_neur  = np.mean(shell_eval_neural) if shell_eval_neural else 0.0

        status = "VICTORIA (SURROGATE >= HUNGARIAN)"
        if mean_fit_neur > mean_fit_hung:
            wins += 1
        elif np.isclose(mean_fit_neur, mean_fit_hung, atol=1e-3):
            ties += 1
            status = "EMPATE (SURROGATE == HUNGARIAN)"
        else:
            losses += 1
            status = "DERROTA (HUNGARIAN > SURROGATE)"

        print(f"\n🧩 [Shell {shell_idx}] -> {status}")
        print(f"   • Baseline (Hungarian) : Fitness Promedio = {mean_fit_hung:.2f} | Evaluaciones Promedio = {mean_ev_hung:,.0f}")
        print(f"   • Surrogate (Production): Fitness Promedio = {mean_fit_neur:.2f} | Evaluaciones Promedio = {mean_ev_neur:,.0f}")
        
        summary_data.append({
            "shell": shell_idx,
            "hungarian_mean_fitness": mean_fit_hung,
            "neural_mean_fitness": mean_fit_neur,
            "hungarian_mean_evals": mean_ev_hung,
            "neural_mean_evals": mean_ev_neur,
            "outcome": status
        })

    # CRITERIO DE ÉXITO PRE-REGISTRADO
    success_rate = ((wins + ties) / total_shells) * 100.0
    print("\n" + "="*80)
    print(f"🎯 RESULTADO DEL CRITERIO PRE-REGISTRADO: {success_rate:.1f}% de éxito en combinaciones por Shell (≥70% requerido)")
    if success_rate >= 70.0:
        print("✅ EL EXPERIMENTO CONFIRMA ÉXITO PRE-REGISTRADO DEL SURROGATE ARCHITECTURE.")
    else:
        print("⚠️ EL EXPERIMENTO NO ALCANZÓ EL UMBRAL DEL 70% PRE-REGISTRADO.")
    print("="*80)

    # Guardar tablas de resultados
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary_table.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "ablation_summary_table.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Generación de Gráficas de Publicación
    print("\nGenerando gráficas de publicación (Fitness vs. Tiempo y Evaluaciones vs. Tiempo)...")
    time_grid = np.linspace(0, TIME_LIMIT_SEC, 200)
    colors = {"hungarian": "#1f77b4", "neural": "#2ca02c"}
    labels = {"hungarian": "ES + A* Exacto (Hungarian, CPU)", "neural": "ES + Surrogate (Producción, GPU + Híbrido)"}
    styles = {"hungarian": "--", "neural": "-"}

    for shell_idx in shells:
        agg_fitness = {"hungarian": [], "neural": []}
        agg_evals   = {"hungarian": [], "neural": []}

        for heur in ["hungarian", "neural"]:
            for seed in seeds:
                csv_file = os.path.join(OUTPUT_DIR, f"ES_{heur}_shell{shell_idx}_seed{seed}_log.csv")
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file, on_bad_lines='skip')
                        df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
                        df = df.dropna(subset=['time_ms', 'fitness', 'evaluations'])
                        if len(df) >= 2:
                            df['time_sec'] = df['time_ms'] / 1000.0
                            interp_fit = np.interp(time_grid, df['time_sec'], df['fitness'])
                            interp_eval = np.interp(time_grid, df['time_sec'], df['evaluations'])
                            agg_fitness[heur].append(interp_fit)
                            agg_evals[heur].append(interp_eval)
                    except:
                        pass

        # 1. Gráfica de Fitness vs. Tiempo
        plt.figure(figsize=(10, 6))
        for heur in ["hungarian", "neural"]:
            if len(agg_fitness[heur]) > 0:
                mean_fit = np.mean(agg_fitness[heur], axis=0)
                std_fit = np.std(agg_fitness[heur], axis=0)
                plt.plot(time_grid, mean_fit, label=labels[heur], color=colors[heur], linestyle=styles[heur], linewidth=2.5)
                plt.fill_between(time_grid, mean_fit - std_fit, mean_fit + std_fit, color=colors[heur], alpha=0.2)

        plt.title(f"Evolución del Fitness vs. Tiempo — Shell {shell_idx} (n=10 semillas)")
        plt.xlabel("Tiempo de Ejecución en Circuito (segundos)")
        plt.ylabel("Dificultad Alcanzada (Fitness FO1: Empujes)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fitness_vs_time_shell_{shell_idx}.pdf"))
        plt.close()

        # 2. Gráfica de Evaluaciones vs. Tiempo
        plt.figure(figsize=(10, 6))
        for heur in ["hungarian", "neural"]:
            if len(agg_evals[heur]) > 0:
                mean_ev = np.mean(agg_evals[heur], axis=0)
                std_ev = np.std(agg_evals[heur], axis=0)
                lower_bound = np.maximum(mean_ev - std_ev, 1)
                upper_bound = mean_ev + std_ev
                plt.plot(time_grid, mean_ev, label=labels[heur], color=colors[heur], linestyle=styles[heur], linewidth=2.5)
                plt.fill_between(time_grid, lower_bound, upper_bound, color=colors[heur], alpha=0.2)

        plt.title(f"Velocidad de Exploración: Evaluaciones vs. Tiempo — Shell {shell_idx} (n=10 semillas)")
        plt.xlabel("Tiempo de Ejecución en Circuito (segundos)")
        plt.ylabel("Nodos Evaluados Acumulados (escala logarítmica)")
        plt.yscale("log")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"evals_vs_time_shell_{shell_idx}.pdf"))
        plt.close()

    print(f"📈 Gráficos PDF y reportes guardados exitosamente en la carpeta `{OUTPUT_DIR}/`.")

def main():
    seeds = [str(i) for i in range(42, 52)]  # 10 semillas (42-51)
    shells = [1, 2, 3, 4, 5]               # 5 shells (1-5)
    heuristics = ["hungarian", "neural"]   # 2 configuraciones (Baseline vs Surrogate)
    
    tasks = [(h, s, sh) for h in heuristics for sh in shells for s in seeds]
    total_tasks = len(tasks)
    
    print(f"🚀 INICIANDO BATERÍA DE ABLACIÓN DE 100 CORRIDAS ({len(heuristics)} configs × {len(shells)} shells × {len(seeds)} semillas)...")
    print(f"⚡ Ejecución en paralelo con 6 hilos concurrentes para aprovechar GPU RTX 5070 Ti y CPU multinúcleo...")
    
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_task = {executor.submit(run_single_experiment, "ES", h, s, sh): (h, s, sh) for (h, s, sh) in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            h, s, sh = future_to_task[future]
            completed += 1
            try:
                res = future.result()
                heur_name, seed_val, shell_val, is_cached, out_txt, disyuntor, deleg, gens, evals, best_fit, elaps = res
                if is_cached:
                    print(f"[{completed:03d}/{total_tasks:03d}] [CACHED] shell_{shell_val}.sok | Config={heur_name:9s} | Seed {seed_val}")
                else:
                    print(f"[{completed:03d}/{total_tasks:03d}] [COMPLETE] shell_{shell_val}.sok | Config={heur_name:9s} | Seed {seed_val} | Fit={best_fit:.1f} | Evals={evals:,} | Triggers={disyuntor} | Deleg={deleg} | Time={elaps:.1f}s", flush=True)
            except Exception as exc:
                print(f"[{completed:03d}/{total_tasks:03d}] [ERROR] shell_{sh}.sok | Config={h:9s} | Seed {s} generó una excepción: {exc}", flush=True)

    # Proceder a análisis y gráficos
    analyze_and_plot(seeds, shells)

if __name__ == "__main__":
    main()
