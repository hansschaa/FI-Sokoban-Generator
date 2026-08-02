import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

OUTPUT_DIR = "experiment_1_matrix_results"
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

# Las 4 Variantes de la Matriz 2x2: (Verificador de Jugabilidad x Decisor de Fitness)
VARIANTS = [
    ("A* Puro", "hungarian"),
    ("Clasificador + A*", "classifier_filter"),
    ("A* verifica + Regresor", "hybrid_regressor"),
    ("Full Surrogate (100% Neural)", "full_surrogate")
]

def set_server_threshold(threshold):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            return True
    except Exception as e:
        return False

def verify_board_with_astar(board_flat_str, tag="exp1"):
    if not board_flat_str or len(board_flat_str) < 5:
        return 0, False
    rows = [r for r in board_flat_str.split("|") if r.strip() != ""]
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{tag}.sok")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
        
    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin): return 0, False

    cmd = [solver_bin, temp_file, "0", "500"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=12, text=True)
    except Exception as e:
        out = getattr(e, "output", "")
        if isinstance(out, bytes): out = out.decode('utf-8', errors='replace')

    pushes = 0
    for line in out.split("\n"):
        if "Pushes:" in line:
            try: pushes = int(line.split(":")[1].strip())
            except: pass
            
    if os.path.exists(temp_file):
        try: os.remove(temp_file)
        except: pass
        
    return pushes, (pushes > 0)

def run_experiment_run(label, heuristic, shell_idx, seed):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_prefix = os.path.join(OUTPUT_DIR, f"{heuristic}_shell{shell_idx}_seed{seed}")
    out_csv = out_prefix + ".csv"
    out_txt = out_prefix + ".txt"
    out_meta = out_prefix + "_meta.json"
    tmp_csv = out_csv + ".tmp"

    if os.path.exists(tmp_csv):
        try: os.remove(tmp_csv)
        except: pass

    if os.path.exists(out_meta):
        try:
            with open(out_meta, "r") as f:
                data = json.load(f)
            return True, data
        except: pass

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path): runner_path = "./build2/experiment_runner"
    if not os.path.exists(runner_path):
        print("❌ Error: No se encontró experiment_runner")
        sys.exit(1)

    th = THRESHOLD_MAP.get(shell_idx, 0.70)
    if heuristic != "hungarian":
        set_server_threshold(th)

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]

    print(f"🚀 [Exp 1] Shell {shell_idx} | Seed {seed:<4} | Var: {label:<24} (Th={th:.2f})...", end=" ", flush=True)

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

    if os.path.exists(tmp_csv): os.rename(tmp_csv, out_csv)
    with open(out_txt, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write(out_text)

    gens = 0; evals = 0; deadlocks_filtered = 0; fp_count = 0; hybrid_del = 0; reg_calls = 0
    neural_fitness = 0.0; best_board = ""
    top_boards = []

    for line in out_text.split('\n'):
        if "[ES STATS] Classifier Deadlocks Filtered" in line:
            try: deadlocks_filtered = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Classifier False Positives" in line:
            try: fp_count = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Surrogate Regressor Calls" in line:
            try: reg_calls = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Hybrid Hungarian Delegations" in line:
            try: hybrid_del = int(line.split(":")[1].strip())
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
        elif line.startswith("RANK_"):
            parts = line.split(";")
            if len(parts) >= 3:
                try:
                    r_lbl = parts[0].strip()
                    n_fit = float(parts[1].strip())
                    b_str = parts[2].strip()
                    if n_fit > -1e8: top_boards.append((r_lbl, n_fit, b_str))
                except: pass
        elif line.strip() and ";" in line.strip() and not line.startswith("RANK_") and not line.startswith("["):
            parts = line.strip().split(";")
            if len(parts) >= 3:
                try: 
                    neural_fitness = -float(parts[0].strip())
                    best_board = parts[2].strip()
                except: pass

    if neural_fitness <= -1e8 or neural_fitness >= 1e8: neural_fitness = 0.0
    if not top_boards and best_board:
        top_boards.append(("RANK_1", neural_fitness, best_board))

    # Auditoría Post-hoc con A* Real
    solvable_top5 = 0
    best_real_astar_pushes = 0
    for r_lbl, n_fit, b_str in top_boards:
        real_p, is_sol = verify_board_with_astar(b_str, tag=f"{heuristic}_sh{shell_idx}_s{seed}")
        if is_sol:
            solvable_top5 += 1
            if real_p > best_real_astar_pushes:
                best_real_astar_pushes = real_p

    tpr_top5_pct = (solvable_top5 / len(top_boards) * 100.0) if top_boards else 0.0
    astar_evals = evals - deadlocks_filtered if heuristic == "classifier_filter" else (evals if heuristic == "hungarian" else hybrid_del)

    unique_boards_count = 0
    if os.path.exists(out_csv):
        try:
            df_traj = pd.read_csv(out_csv, on_bad_lines='skip')
            if 'best_board' in df_traj.columns:
                unique_boards_count = int(df_traj['best_board'].nunique())
        except: pass

    meta_record = {
        "Variant": label,
        "Heuristic": heuristic,
        "Shell_Idx": shell_idx,
        "Shell": f"Shell {shell_idx}",
        "Seed": seed,
        "Threshold": th,
        "Generations": gens,
        "Total_Evals": evals,
        "Unique_Boards_Explored": unique_boards_count,
        "Deadlocks_Filtered": deadlocks_filtered,
        "False_Positives": fp_count,
        "Hybrid_Delegations_6PlusBoxes": hybrid_del,
        "Regressor_Calls": reg_calls,
        "Astar_Evals_In_Loop": astar_evals,
        "Top1_Neural_Fit": round(neural_fitness, 1),
        "Top5_Best_Real_Astar_Pushes": best_real_astar_pushes,
        "Top5_Solvable_Count": solvable_top5,
        "Top5_Size": len(top_boards),
        "Top5_Accuracy_Pct": round(tpr_top5_pct, 1),
        "Time_s": round(elapsed, 1)
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2)

    print(f"✔️ Done ({elapsed:.1f}s) | A* Real Best: {best_real_astar_pushes} | Top-5 Soluble: {solvable_top5}/{len(top_boards)} ({tpr_top5_pct:.0f}%) | Tableros Únicos: {unique_boards_count}")
    return False, meta_record

def generate_final_analysis():
    print("\n📊 Agregando telemetría y generando figuras de publicación...")
    records = []
    for (lbl, heur) in VARIANTS:
        for sh in SHELLS:
            for s in SEEDS:
                meta_path = os.path.join(OUTPUT_DIR, f"{heur}_shell{sh}_seed{s}_meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            data = json.load(f)
                        records.append(data)
                    except: pass

    if not records:
        print("⚠️ No hay registros terminados.")
        return

    df = pd.DataFrame(records)
    df_csv = os.path.join(OUTPUT_DIR, "experiment_1_complete_runs.csv")
    df.to_csv(df_csv, index=False)
    print(f"📁 Todas las corridas exportadas a: {df_csv}")

    # Resumen por Shell y Variante
    cols_to_avg = ["Top5_Best_Real_Astar_Pushes", "Top5_Accuracy_Pct", "Time_s", "Unique_Boards_Explored", "Hybrid_Delegations_6PlusBoxes", "Deadlocks_Filtered"]
    if "Top5_Inconclusive_Count" in df.columns:
        cols_to_avg.insert(2, "Top5_Inconclusive_Count")
    summary = df.groupby(["Shell", "Variant"])[cols_to_avg].mean().reset_index().round(1)

    print("\n" + "="*130)
    print(" 🏆 EXPERIMENTO 1: MATRIZ 2x2 COMPRENSIVA (MEDIA SOBRE 10 SEMILLAS POR CONFIGURACIÓN | 300s POR CORRIDA)")
    print("="*130)
    print(summary.to_string(index=False))

    sum_path = os.path.join(OUTPUT_DIR, "experiment_1_summary_table.csv")
    summary.to_csv(sum_path, index=False)
    print(f"📁 Tabla resumen guardada en: {sum_path}")

    # Gráficas
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Shell", y="Top5_Best_Real_Astar_Pushes", hue="Variant", palette="Set2")
    plt.title("Experimento 1: Calidad Certificada por A* (Pushes Reales en Top-5 Final)")
    plt.ylabel("Pushes Reales (A* Ground Truth)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp1_real_pushes_boxplot.pdf"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Shell", y="Time_s", hue="Variant", palette="mako")
    plt.title("Tiempo Computacional hasta Convergencia / Límite en Matriz 2x2")
    plt.ylabel("Tiempo Promedio (segundos)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp1_execution_time_barplot.pdf"))
    plt.close()
    print("📈 Figuras PDF guardadas en la carpeta experiment_1_matrix_results/.")

def main():
    print("\n" + "="*120)
    print(" 🧪 EXPERIMENTO 1 DEFINITIVO: MATRIZ 2x2 ESTUDIO DE ABLACIÓN GENERAL (5 SHELLS x 10 SEEDS x 300s)")
    print(f" Política de Umbral por Escasez Estructural: {THRESHOLD_MAP}")
    print("="*120)

    total_runs = len(VARIANTS) * len(SHELLS) * len(SEEDS)
    count = 0
    for (lbl, heur) in VARIANTS:
        for sh in SHELLS:
            for s in SEEDS:
                count += 1
                print(f"[{count:03d}/{total_runs:03d}] ", end="")
                run_experiment_run(lbl, heur, sh, s)

    set_server_threshold(0.70)
    generate_final_analysis()

if __name__ == "__main__":
    main()
