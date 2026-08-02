import os
import subprocess
import time
import sys
import pandas as pd
import numpy as np
import json
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14})

OUTPUT_DIR = "ablation_classifier_filter_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 380
SERVER_URL = "http://127.0.0.1:5000"

# Política de Umbral por Shell (0.60 para Shell 3 por escasez estructural, 0.70 para el resto)
THRESHOLD_MAP = {
    1: 0.70,
    2: 0.70,
    3: 0.60,
    4: 0.70,
    5: 0.70
}

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
SHELLS = [1, 2, 3, 4, 5]

def set_server_threshold(threshold):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            return True
    except Exception as e:
        print(f"⚠️ Aviso: No se pudo actualizar umbral en el servidor neuronal ({e}). Si estás corriendo sin clasificador no hay problema.")
        return False

def run_experiment(algo, heuristic_label, shell_idx, seed):
    heuristic = "hungarian" if heuristic_label == "sin_clasificador" else "classifier_filter"
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = os.path.join(OUTPUT_DIR, f"{heuristic_label}_shell{shell_idx}_seed{seed}.csv")
    out_txt = os.path.join(OUTPUT_DIR, f"{heuristic_label}_shell{shell_idx}_seed{seed}.txt")
    tmp_csv = out_csv + ".tmp"
    
    if os.path.exists(tmp_csv):
        try: os.remove(tmp_csv)
        except: pass

    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0 and os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_txt, "r", encoding="utf-8", errors="replace") as f:
            out_text = f.read()
        best_fit = 0.0
        gens = 0
        evals = 0
        df_csv = pd.read_csv(out_csv, on_bad_lines='skip')
        if len(df_csv) > 0 and 'fitness' in df_csv.columns:
            best_fit = float(df_csv['fitness'].iloc[-1])
            if 'evaluations' in df_csv.columns: evals = int(df_csv['evaluations'].iloc[-1])
        if best_fit <= -1e8 or best_fit >= 1e8: best_fit = 0.0
        return True, best_fit, evals

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path):
        runner_path = "./build2/experiment_runner"

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]

    th = THRESHOLD_MAP.get(shell_idx, 0.70) if heuristic == "classifier_filter" else 0.70
    if heuristic == "classifier_filter":
        set_server_threshold(th)

    print(f"🚀 Shell {shell_idx} | Seed {seed:<4} | Config: {heuristic_label:<18} (Th={th}) | Límite: {TIME_LIMIT_SEC}s...", end=" ", flush=True)

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes): out_text = out_text.decode('utf-8', errors='replace')
    elapsed = time.time() - start_time

    if os.path.exists(tmp_csv):
        os.rename(tmp_csv, out_csv)
    with open(out_txt, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write(out_text)

    best_fitness = 0.0
    evals = 0
    gens = 0
    deadlocks_filtered = 0
    false_positives = 0

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

    if best_fitness == 0.0 and os.path.exists(out_csv):
        try:
            df_csv = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df_csv) > 0 and 'fitness' in df_csv.columns:
                best_fitness = float(df_csv['fitness'].iloc[-1])
                if evals == 0 and 'evaluations' in df_csv.columns: evals = int(df_csv['evaluations'].iloc[-1])
        except: pass

    if best_fitness <= -1e8 or best_fitness >= 1e8:
        best_fitness = 0.0

    print(f"✔️ Done! ({elapsed:.1f}s) | Fit: {best_fitness} | Gens: {gens} | Evals: {evals} | Filtrados: {deadlocks_filtered} | FP: {false_positives}")
    return False, best_fitness, evals

def generate_reports_and_plots():
    print("\n📊 Generando tablas resumen y gráficas de ablación...")
    records = []
    
    for shell in SHELLS:
        for seed in SEEDS:
            for conf in ["sin_clasificador", "con_clasificador"]:
                out_csv = os.path.join(OUTPUT_DIR, f"{conf}_shell{shell}_seed{seed}.csv")
                out_txt = os.path.join(OUTPUT_DIR, f"{conf}_shell{shell}_seed{seed}.txt")
                if not os.path.exists(out_txt) or not os.path.exists(out_csv):
                    continue
                
                with open(out_txt, "r", encoding="utf-8", errors="replace") as f:
                    out_text = f.read()
                
                gens = 0; evals = 0; df_count = 0; fp = 0; best_fit = 0.0
                for line in out_text.split('\n'):
                    if "[ES STATS] Classifier Deadlocks Filtered" in line:
                        try: df_count = int(line.split(":")[1].strip())
                        except: pass
                    elif "[ES STATS] Classifier False Positives" in line:
                        try: fp = int(line.split(":")[1].strip())
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
                            try: best_fit = -float(parts[0].strip())
                            except: pass
                            
                if best_fit == 0.0:
                    try:
                        df_c = pd.read_csv(out_csv, on_bad_lines='skip')
                        if len(df_c) > 0 and 'fitness' in df_c.columns:
                            best_fit = float(df_c['fitness'].iloc[-1])
                            if evals == 0 and 'evaluations' in df_c.columns: evals = int(df_c['evaluations'].iloc[-1])
                    except: pass
                if best_fit <= -1e8 or best_fit >= 1e8: best_fit = 0.0
                
                astar_evals = evals - df_count if conf == "con_clasificador" else evals
                records.append({
                    "Shell": f"Shell {shell}",
                    "Seed": seed,
                    "Config": "Con Clasificador (Th Adaptativo)" if conf == "con_clasificador" else "Sin Clasificador (A*)",
                    "Fitness": best_fit,
                    "Generations": gens,
                    "Total_Evals": evals,
                    "Deadlocks_Filtered": df_count,
                    "False_Positives": fp,
                    "Astar_Evals": astar_evals
                })
                
    if not records:
        print("⚠️ No hay suficientes datos procesados para generar tablas.")
        return

    df = pd.DataFrame(records)
    summary = df.groupby(["Shell", "Config"])[["Fitness", "Generations", "Astar_Evals", "Deadlocks_Filtered", "False_Positives"]].mean().reset_index()
    summary = summary.round(1)
    
    print("\n" + "="*105)
    print(" 🏆 RESUMEN GENERAL DE LA ABLACIÓN COMPROMETIDA (10 SEMILLAS POR SHELL | 300s POR CORRIDA)")
    print("="*105)
    print(summary.to_string(index=False))
    
    summary_path = os.path.join(OUTPUT_DIR, "ablation_summary_table.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n📁 Tabla resumen guardada en: {summary_path}")

    # Gráficas
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Shell", y="Fitness", hue="Config", palette="Set2")
    plt.title("Estudio de Ablación Completo: Impacto del Filtro Neuronal en Calidad del Tablero (Fitness)")
    plt.ylabel("Pushes Máximos (Fitness FO1)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_fitness_boxplot.pdf"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Shell", y="Astar_Evals", hue="Config", palette="mako")
    plt.title("Reducción de Evaluaciones Costosas (A* Reales) por Pre-filtrado Neuronal")
    plt.ylabel("Evaluaciones A* Ejecutadas")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_astar_savings_barplot.pdf"))
    plt.close()
    print("📈 Gráficas comparativas guardadas exitosamente como PDF en la carpeta de resultados.")

def main():
    print("\n" + "="*105)
    print(" 🚀 ESTUDIO DE ABLACIÓN GENERAL: FILTRO NEURAL CON POLÍTICA DE UMBRAL POR ESCASEZ ESTRUCTURAL")
    print(f" Mapeo de Umbrales: {THRESHOLD_MAP}")
    print("="*105)
    
    total_runs = len(SHELLS) * len(SEEDS) * 2
    count = 0
    for shell in SHELLS:
        for seed in SEEDS:
            for conf in ["sin_clasificador", "con_clasificador"]:
                count += 1
                print(f"[{count:02d}/{total_runs:02d}] ", end="")
                run_experiment("ES", conf, shell, seed)
                
    # Volver a umbral 0.70 de producción por seguridad
    set_server_threshold(0.70)
    generate_reports_and_plots()

if __name__ == "__main__":
    main()
