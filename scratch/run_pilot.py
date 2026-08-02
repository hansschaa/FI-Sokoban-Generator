import os
import subprocess
import time
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual para publicación
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

OUTPUT_DIR = "pilot_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300  # 5 minutos de tiempo de circuito por corrida
PYTHON_TIMEOUT = 480  # Protección de 8 minutos para permitir cierre elegante del C++

def run_single_experiment(algo, heuristic, seed, shell_idx):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = os.path.join(OUTPUT_DIR, f"{algo}_{heuristic}_shell{shell_idx}_seed{seed}_log.csv")
    out_txt_file = os.path.join(OUTPUT_DIR, f"{algo}_{heuristic}_shell{shell_idx}_seed{seed}_stdout.txt")
    tmp_csv = out_csv + ".tmp"
    
    if os.path.exists(tmp_csv):
        try: os.remove(tmp_csv)
        except: pass
            
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        best_fitness = -1e9
        evals = 0
        try:
            df = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df) > 0 and 'fitness' in df.columns:
                best_fitness = float(df['fitness'].iloc[-1])
                if 'evaluations' in df.columns: evals = int(df['evaluations'].iloc[-1])
        except: pass
        return (heuristic, seed, shell_idx, True, "Already executed", 0, 0, 0, evals, best_fitness, 1, 0.0)

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
    with open(out_txt_file, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write(out_text)

    disyuntor_count = 0
    delegations = 0
    gens = 0
    evals = 0
    init_attempts = 1
    best_fitness = -1e9

    for line in out_text.split('\n'):
        if "[ES STATS] Circuit Breaker (MAX_FAILURES) triggers:" in line:
            try: disyuntor_count = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Hybrid Hungarian Delegations (box_count >= 6):" in line:
            try: delegations = int(line.split(":")[1].strip())
            except: pass
        elif "[INIT STATS] Initial seed found in" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "in" and i+1 < len(parts):
                    try: init_attempts = int(parts[i+1])
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
            parts = line.strip().split(";")
            if len(parts) >= 3:
                try: best_fitness = -float(parts[0].strip())
                except: pass

    if best_fitness == -1e9 and os.path.exists(out_csv):
        try:
            df = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df) > 0 and 'fitness' in df.columns:
                best_fitness = float(df['fitness'].iloc[-1])
                if evals == 0 and 'evaluations' in df.columns:
                    evals = int(df['evaluations'].iloc[-1])
        except: pass

    return (heuristic, seed, shell_idx, False, out_text, disyuntor_count, delegations, gens, evals, best_fitness, init_attempts, elapsed)

def get_fitness_at_time(df, target_sec):
    if len(df) == 0:
        return 0.0
    target_ms = target_sec * 1000.0
    sub_df = df[df['time_ms'] <= target_ms]
    if len(sub_df) > 0:
        return float(sub_df['fitness'].iloc[-1])
    else:
        # Si ni siquiera había empezado la primera generación al momento target_sec, devolver fitness inicial o 0
        return float(df['fitness'].iloc[0])

def analyze_and_plot(seeds, shells):
    print("\n" + "="*85)
    print(" 📊 ANÁLISIS DE ABLACIÓN PRE-REGISTRADO: CORTES DE TIEMPO Y DIVERSIDAD ESTRUCTURAL")
    print("="*85)

    budget_cuts = [30, 60, 120, 300]
    results_by_cut = {c: [] for c in budget_cuts}
    diversity_stats = []
    
    total_shells = len(shells)

    for shell_idx in shells:
        shell_dfs = {"hungarian": [], "neural": []}
        for heur in ["hungarian", "neural"]:
            for seed in seeds:
                csv_path = os.path.join(OUTPUT_DIR, f"ES_{heur}_shell{shell_idx}_seed{seed}_log.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, on_bad_lines='skip')
                        df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
                        df = df.dropna(subset=['time_ms', 'fitness'])
                        shell_dfs[heur].append(df)
                    except: pass

        # 1. Análisis de Diversidad Estructual (Número de "Atractores" / Tableros únicos visitados)
        div_hung, div_neur = [], []
        for df in shell_dfs["hungarian"]:
            if 'best_board' in df.columns:
                div_hung.append(len(df['best_board'].dropna().unique()))
            else:
                div_hung.append(1)
        for df in shell_dfs["neural"]:
            if 'best_board' in df.columns:
                div_neur.append(len(df['best_board'].dropna().unique()))
            else:
                div_neur.append(1)

        mean_div_hung = np.mean(div_hung) if div_hung else 0.0
        mean_div_neur = np.mean(div_neur) if div_neur else 0.0
        diversity_stats.append({
            "shell": shell_idx,
            "hungarian_unique_boards": mean_div_hung,
            "neural_unique_boards": mean_div_neur,
            "diff_percent": ((mean_div_neur - mean_div_hung) / max(mean_div_hung, 1)) * 100.0
        })

        # 2. Análisis a Distintos Presupuestos de Tiempo (30s, 60s, 120s, 300s)
        for cut in budget_cuts:
            fits_hung = [get_fitness_at_time(df, cut) for df in shell_dfs["hungarian"]]
            fits_neur = [get_fitness_at_time(df, cut) for df in shell_dfs["neural"]]
            
            mean_fit_h = np.mean(fits_hung) if fits_hung else 0.0
            mean_fit_n = np.mean(fits_neur) if fits_neur else 0.0
            
            if mean_fit_n > mean_fit_h: status = "VICTORIA"
            elif np.isclose(mean_fit_n, mean_fit_h, atol=1e-3): status = "EMPATE"
            else: status = "DERROTA"

            results_by_cut[cut].append({
                "shell": shell_idx,
                "hungarian_mean_fit": mean_fit_h,
                "neural_mean_fit": mean_fit_n,
                "status": status
            })

    # IMPRIMIR REPORTE POR CORTES DE PRESUPUESTO DE TIEMPO
    print("\n" + "-"*85)
    print(" ⏱️ EVALUACIÓN DEL CRITERIO PRE-REGISTRADO (≥70% VICTORIAS/EMPATES) POR PRESUPUESTO")
    print("-" * 85)
    
    cut_summary_export = {}
    for cut in budget_cuts:
        wins_cut = sum(1 for r in results_by_cut[cut] if r["status"] == "VICTORIA")
        ties_cut = sum(1 for r in results_by_cut[cut] if r["status"] == "EMPATE")
        losses_cut = sum(1 for r in results_by_cut[cut] if r["status"] == "DERROTA")
        succ_rate = ((wins_cut + ties_cut) / max(total_shells, 1)) * 100.0
        
        print(f"\n⌛ PRESUPUESTO = {cut:3d}s | Éxito: {succ_rate:5.1f}% ({wins_cut} Victorias, {ties_cut} Empates, {losses_cut} Derrotas)")
        for r in results_by_cut[cut]:
            print(f"   • Shell {r['shell']}: Baseline (A*) = {r['hungarian_mean_fit']:5.2f} | Surrogate (GPU) = {r['neural_mean_fit']:5.2f} -> {r['status']}")
            
        if succ_rate >= 70.0:
            print(f"   🎯 Conclusión a los {cut}s: CUMPLE criterio pre-registrado del 70%.")
        else:
            print(f"   ⚠️ Conclusión a los {cut}s: NO ALCANZA umbral del 70% (Resultado científico legítimo para reportar).")
        
        cut_summary_export[f"budget_{cut}s"] = {
            "success_rate_pct": succ_rate,
            "wins": wins_cut, "ties": ties_cut, "losses": losses_cut,
            "shell_details": results_by_cut[cut]
        }

    # IMPRIMIR REPORTE DE DIVERSIDAD ESTRUCTURAL
    print("\n" + "-"*85)
    print(" 🎨 EVALUACIÓN DE DIVERSIDAD ESTRUCTURAL (TABLEROS MEJOR-DE-GENERACIÓN ÚNICOS VISITADOS)")
    print("-" * 85)
    for d in diversity_stats:
        print(f"🧩 [Shell {d['shell']}] Promedio de Atractores Explorados por Corrida:")
        print(f"   • Baseline (Hungarian A*): {d['hungarian_unique_boards']:.1f} tableros únicos")
        print(f"   • Surrogate Híbrido (GPU): {d['neural_unique_boards']:.1f} tableros únicos ({d['diff_percent']:+.1f}%)")

    # Guardar reportes en JSON y CSV
    with open(os.path.join(OUTPUT_DIR, "ablation_multibudget_analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"budget_analysis": cut_summary_export, "diversity_analysis": diversity_stats}, f, indent=4)
    pd.DataFrame(diversity_stats).to_csv(os.path.join(OUTPUT_DIR, "ablation_diversity_table.csv"), index=False)

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
                    except: pass

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

    print(f"\n📈 Gráficos PDF y reportes multi-presupuesto y de diversidad guardados exitosamente en `{OUTPUT_DIR}/`.")

def main():
    seeds = ["42"]  # 1 semilla (ultra corto)
    shells = [1, 2, 3]               # 3 shells (ultra corto)
    heuristics = ["hungarian", "neural"]   # 2 configuraciones (Baseline vs Surrogate)
    
    tasks = [(h, s, sh) for h in heuristics for sh in shells for s in seeds]
    total_tasks = len(tasks)
    
    print(f"🚀 INICIANDO PILOTO ULTRA CORTO DE 6 CORRIDAS ({len(heuristics)} configs × {len(shells)} shells × {len(seeds)} semillas)...")
    print(f"🔒 Ejecución ESTRICTAMENTE SECUENCIAL (1 corrida a la vez) para eliminar ruido de concurrencia...")
    
    completed = 0
    for (h, s, sh) in tasks:
        completed += 1
        try:
            res = run_single_experiment("ES", h, s, sh)
            heur_name, seed_val, shell_val, is_cached, out_txt, disyuntor, deleg, gens, evals, best_fit, init_att, elaps = res
            if is_cached:
                print(f"[{completed:03d}/{total_tasks:03d}] [CACHED] shell_{shell_val}.sok | Config={heur_name:9s} | Seed {seed_val}")
            else:
                print(f"[{completed:03d}/{total_tasks:03d}] [COMPLETE] shell_{shell_val}.sok | Config={heur_name:9s} | Seed {seed_val} | Fit={best_fit:.1f} | Evals={evals:,} | InitAtt={init_att} | Triggers={disyuntor} | Deleg={deleg} | Time={elaps:.1f}s", flush=True)
        except Exception as exc:
            print(f"[{completed:03d}/{total_tasks:03d}] [ERROR] shell_{sh}.sok | Config={h:9s} | Seed {s} generó una excepción: {exc}", flush=True)

    # Proceder a análisis multi-presupuesto y gráficos
    analyze_and_plot(seeds, shells)

if __name__ == "__main__":
    main()
