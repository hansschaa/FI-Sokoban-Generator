import os
import subprocess
import time
import pandas as pd
import numpy as np
import sys

OUTPUT_DIR = "pilot_classifier_filter_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300  # 5 minutos por corrida
PYTHON_TIMEOUT = 360

def run_experiment(config_name, heuristic, seed, shell_idx):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = os.path.join(OUTPUT_DIR, f"ES_{config_name}_shell{shell_idx}_seed{seed}.csv")
    out_txt_file = os.path.join(OUTPUT_DIR, f"ES_{config_name}_shell{shell_idx}_seed{seed}.txt")
    tmp_csv = out_csv + ".tmp"
    
    if os.path.exists(tmp_csv):
        try: os.remove(tmp_csv)
        except: pass

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path):
        runner_path = "./build2/experiment_runner"
    if not os.path.exists(runner_path):
        print("❌ Error: No se encontró experiment_runner en ./build/ ni ./build2/")
        sys.exit(1)

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]

    print(f"🚀 Ejecutando {config_name:<18} | Shell {shell_idx} | Semilla {seed} (Límite: {TIME_LIMIT_SEC}s)...")
    
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

    gens = 0
    evals = 0
    deadlocks_filtered = 0
    false_positives = 0
    best_fitness = -1e9

    for line in out_text.split('\n'):
        if "[ES STATS] Classifier Deadlocks Filtered" in line:
            try: deadlocks_filtered = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Classifier False Positives" in line:
            try: false_positives = int(line.split(":")[1].strip())
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

    astar_evals = evals - deadlocks_filtered if config_name == "con_clasificador" else evals
    return {
        "shell": f"shell_{shell_idx}.sok",
        "config": config_name,
        "total_evals": evals,
        "neural_filtered": deadlocks_filtered,
        "false_positives": false_positives,
        "astar_evals": astar_evals,
        "generations": gens,
        "best_fitness": best_fitness if best_fitness != -1e9 else 0.0,
        "time_s": round(elapsed, 1)
    }

def main():
    print("\n" + "="*95)
    print(" 🧪 PILOTO RÁPIDO DE ABLACIÓN: AISLANTE DEL CLASIFICADOR CONTRASTIVO COMO FILTRO PRE-A*")
    print(" Configuración de prueba: 5 Shells x 1 Semilla (Seed 42) x 300s por corrida")
    print("="*95)

    shells = [1, 2, 3, 4, 5]
    seed = 42
    configs = [
        ("sin_clasificador", "hungarian"),
        ("con_clasificador", "classifier_filter")
    ]

    results = []
    for shell_idx in shells:
        print(f"\n--- Evaluando Shell {shell_idx} ---")
        for config_name, heuristic in configs:
            res = run_experiment(config_name, heuristic, seed, shell_idx)
            results.append(res)
            print(f"   👉 Resultado: Fitness={res['best_fitness']} | Gens={res['generations']} | Evals={res['total_evals']} | Filtrados={res['neural_filtered']} | FalsoPositivos={res['false_positives']} | A* Reales={res['astar_evals']} | Tiempo={res['time_s']}s")

    print("\n" + "="*95)
    print(" 📋 TABLA COMPARATIVA DE RESULTADOS DEL PILOTO")
    print("="*95)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Resumen agregado
    sin_df = df[df['config'] == 'sin_clasificador']
    con_df = df[df['config'] == 'con_clasificador']
    
    mean_fit_sin = sin_df['best_fitness'].mean()
    mean_fit_con = con_df['best_fitness'].mean()
    mean_evals_sin = sin_df['total_evals'].mean()
    mean_evals_con = con_df['total_evals'].mean()
    total_filtered = con_df['neural_filtered'].sum()
    total_fp = con_df['false_positives'].sum()
    total_evals_con = con_df['total_evals'].sum()
    
    pct_filtered = (total_filtered / total_evals_con * 100) if total_evals_con > 0 else 0
    # Specificity / Precisión del filtro: de las que pasaron el filtro y A* evaluó como deadlock (FP)
    # Tasa de FP sobre el total de evaluaciones o sobre los aprobados
    pct_fp_total = (total_fp / total_evals_con * 100) if total_evals_con > 0 else 0
    total_approved = total_evals_con - total_filtered
    pct_fp_approved = (total_fp / total_approved * 100) if total_approved > 0 else 0

    print("\n" + "-"*95)
    print(" 🏆 DIAGNÓSTICO AGREGADO DEL PILOTO")
    print(f"  • Fitness Promedio       -> Sin Clasificador: {mean_fit_sin:.2f} | Con Clasificador: {mean_fit_con:.2f} (Delta: {mean_fit_con - mean_fit_sin:+.2f})")
    print(f"  • Exploración Media      -> Sin Clasificador: {mean_evals_sin:.0f} evals | Con Clasificador: {mean_evals_con:.0f} evals")
    print(f"  • Eficiencia Neural      -> Se filtraron {total_filtered:,} deadlocks obvios ({pct_filtered:.1f}% de mutaciones descartadas sin gastar A*).")
    print(f"  • Falsos Positivos (FP)  -> {total_fp:,} mutaciones fueron aprobadas por la red pero rechazadas por A* como deadlock.")
    print(f"                              ({pct_fp_approved:.1f}% de las mutaciones que entraron a A* resultaron ser falsos positivos del modelo).")
    print("-" * 95 + "\n")

if __name__ == "__main__":
    main()
