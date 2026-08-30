import os
import subprocess
import time
import json
import hashlib
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

OUTPUT_DIR = "experiment_1_matrix_results_neural_update"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 380
SERVER_URL = "http://127.0.0.1:5000"

# Política de Umbral por Shell (0.60 para Shell 3 por escasez estructural, 0.70 para el resto)
THRESHOLD_MAP = {
    1: 0.90,
    2: 0.90,
    3: 0.85,
    4: 0.90,
    5: 0.90
}

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SHELLS = [1, 2, 3, 4, 5]
CORES = [24]  # Solo variante paralela para ahorrar tiempo

VARIANTS = [
    ("A* verifica + Regresor (Baseline)", "hybrid_regressor", "FO1"),
    ("Full Surrogate Top-K=5", "full_surrogate", "FO1")
]

# ─── HASH DE PIPELINE ──────────────────────────────────────────────────────────
# Embebe un hash del contenido de los componentes críticos del pipeline en cada
# meta.json. Un meta sólo se reutiliza si su hash coincide con el actual.
# Esto es robusto a git pull entre máquinas (mtime es NO confiable cross-machine).
PIPELINE_HASH_FILES = [
    "./build/experiment_runner",                            # binario C++ recompilado
    "./build2/experiment_runner",                           # binario alternativo
    "surrogate_models/surrogate_server.py",                 # server Python
    "surrogate_models/results/regressor_calibration.json",  # calibración C++
    "src/neural_heuristic.cpp",                             # fuente C++ con fix log1p
    "surrogate_models/results/surrogate_stats.txt",         # estadísticos de normalización
    "scratch/run_exp1_neural_update.py",                    # para invalidar cache si el script cambia (ej. umbral)
]

def compute_pipeline_hash():
    h = hashlib.sha256()
    for path in PIPELINE_HASH_FILES:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    h.update(path.encode())
                    h.update(f.read())
            except Exception:
                pass
    return h.hexdigest()[:16]  # 16 hex chars son más que suficientes

CURRENT_PIPELINE_HASH = compute_pipeline_hash()
# ───────────────────────────────────────────────────────────────────────────────

def set_server_threshold(threshold, retries=3):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                json.loads(resp.read().decode('utf-8'))
                return True
        except Exception as e:
            print(f"\n⚠️  set_server_threshold({threshold}) intento {attempt+1}/{retries} falló: {e}")
            time.sleep(2)
    print(f"\n❌ CRÍTICO: No se pudo confirmar umbral={threshold} en el server tras {retries} intentos. Abortando para evitar corrida con umbral incorrecto.")
    sys.exit(1)

# Retorna (pushes, is_solved, is_inconclusive)
# is_inconclusive=True significa que A* no terminó a tiempo — NO es un deadlock confirmado
def verify_board_with_astar(board_flat_str, tag="exp1"):
    if not board_flat_str or len(board_flat_str) < 5:
        return 0, False, False
    rows = [r for r in board_flat_str.split("|") if r.strip() != ""]
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{tag}.sok")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
        
    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin): return 0, False, True  # no solver = inconcluso

    # 60s timeout: distingue "no terminó a tiempo" (INCONCLUSIVE) de "deadlock confirmado" (DEADLOCK)
    cmd = [solver_bin, temp_file, "0", "500"]
    timed_out = False
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60, text=True)
    except subprocess.TimeoutExpired as e:
        out = e.output if e.output else ""
        if isinstance(out, bytes): out = out.decode('utf-8', errors='replace')
        timed_out = True
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

    if pushes > 0:
        return pushes, True, False   # SOLVED
    elif timed_out:
        return 0, False, True        # INCONCLUSIVE (timeout, no es deadlock confirmado)
    else:
        return 0, False, False       # DEADLOCK genuino (A* terminó sin solución)

def run_experiment_run(label, heuristic, fo_type, shell_idx, seed, cores):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_prefix = os.path.join(OUTPUT_DIR, f"{heuristic}_{fo_type}_shell{shell_idx}_seed{seed}_cores{cores}")
    out_csv = out_prefix + ".csv"
    out_txt = out_prefix + ".txt"
    out_meta = out_prefix + "_meta.json"
    tmp_csv = out_csv + ".tmp"

    if os.path.exists(tmp_csv):
        try: os.remove(tmp_csv)
        except: pass

    # VALIDACIÓN DE CACHE: Un meta es válido SOLO si su pipeline_hash coincide con el actual.
    # Esto es robusto a git pull entre máquinas (mtime cross-machine NO es confiable).
    if os.path.exists(out_meta):
        try:
            with open(out_meta, "r") as f:
                data = json.load(f)
            stored_hash = data.get("pipeline_hash", "MISSING")
            if stored_hash == CURRENT_PIPELINE_HASH:
                return True, data  # ✅ Cache válido: mismo pipeline
            else:
                print(f"⚠️  Meta obsoleto (hash={stored_hash[:8]}… != {CURRENT_PIPELINE_HASH[:8]}…) — pipeline cambió. Re-corriendo.")
        except: pass

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path): runner_path = "./build2/experiment_runner"
    if not os.path.exists(runner_path):
        print("❌ Error: No se encontró experiment_runner")
        sys.exit(1)

    th = THRESHOLD_MAP.get(shell_idx, 0.70)


    cmd = [
        runner_path, "ES", fo_type, str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]
    
    if fo_type == "FO6":
        cmd.extend(["--mu", "7", "--lambda", "64", "--mutRate", "0.9381", "--stagLimit", "456"])
    elif heuristic == "full_surrogate":
        cmd.extend(["--mu", "10", "--lambda", "126", "--mutRate", "0.7135", "--stagLimit", "899"])
    else:
        cmd.extend(["--mu", "9", "--lambda", "28", "--mutRate", "0.8559", "--stagLimit", "200"])

    if cores == 1:
        cmd.append("--no-parallel")

    print(f"🚀 [Exp 1] Shell {shell_idx} | Seed {seed:<4} | Cores {cores:<2} | Var: {label:<24} (Th={th:.2f})...", flush=True)

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    termination_reason = "UNKNOWN"
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes): out_text = out_text.decode('utf-8', errors='replace')
        termination_reason = "PYTHON_TIMEOUT"
    elapsed = time.time() - start_time

    if os.path.exists(tmp_csv): os.rename(tmp_csv, out_csv)
    with open(out_txt, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write(out_text)

    gens = 0; evals = 0; deadlocks_filtered = 0; fp_count = 0; hybrid_del = 0; reg_calls = 0
    neural_fitness = 0.0; best_board = ""
    top_boards = []

    for line in out_text.split('\n'):
        if "[ES] Criterio de Parada Alcanzado:" in line:
            if "TIME LIMIT" in line:
                termination_reason = "TIME_LIMIT"
            elif "STAGNATION" in line:
                termination_reason = "STAGNATION"
            elif "MAX_EVALUATIONS" in line:
                termination_reason = "MAX_EVALUATIONS"
        elif "[ES STATS] Classifier Deadlocks Filtered" in line:
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

    # Auditoría Post-hoc con A* Real (con distinción INCONCLUSIVE vs DEADLOCK)
    solvable_top5 = 0
    inconclusive_top5 = 0
    deadlock_top5 = 0
    best_real_astar_pushes = 0
    for r_lbl, n_fit, b_str in top_boards:
        real_p, is_sol, is_inconclusive = verify_board_with_astar(b_str, tag=f"{heuristic}_{fo_type}_sh{shell_idx}_s{seed}_c{cores}")
        if is_sol:
            solvable_top5 += 1
            if real_p > best_real_astar_pushes:
                best_real_astar_pushes = real_p
        elif is_inconclusive:
            inconclusive_top5 += 1
        else:
            deadlock_top5 += 1

    # Precisión solo sobre casos definitivos (excluye inconclusos de numerador y denominador)
    definite_total = solvable_top5 + deadlock_top5
    tpr_top5_pct = (solvable_top5 / definite_total * 100.0) if definite_total > 0 else 0.0
    astar_evals = evals - deadlocks_filtered if heuristic == "classifier_filter" else (evals if heuristic == "hungarian" else hybrid_del)

    unique_boards_count = 0
    is_collapsed = 0
    if os.path.exists(out_csv):
        try:
            df_traj = pd.read_csv(out_csv, on_bad_lines='skip')
            if 'best_board' in df_traj.columns:
                unique_boards_count = int(df_traj['best_board'].nunique())
                if len(df_traj) > 0 and len(df_traj) <= 35:
                    if df_traj.iloc[0]['best_board'] == df_traj.iloc[-1]['best_board']:
                        is_collapsed = 1
        except: pass

    meta_record = {
        "Variant": label,
        "Heuristic": heuristic,
        "FO_Type": fo_type,
        "Shell_Idx": shell_idx,
        "Shell": f"Shell {shell_idx}",
        "Seed": seed,
        "Cores": cores,
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
        "Top5_Inconclusive_Count": inconclusive_top5,
        "Top5_Deadlock_Count": deadlock_top5,
        "Top5_Size": len(top_boards),
        "Top5_Accuracy_Pct": round(tpr_top5_pct, 1),
        "Time_s": round(elapsed, 1),
        "Termination_Reason": termination_reason,
        "Collapsed_Immediate": is_collapsed,
        "pipeline_hash": CURRENT_PIPELINE_HASH   # Firmado con hash del pipeline — invalida la cache si el pipeline cambia
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_record, f, indent=2)

    print(f"✔️  [Shell {shell_idx} | Seed {seed:<4} | {label:<24}] Done ({elapsed:.1f}s, {termination_reason}) | A* Real Best: {best_real_astar_pushes} | Top-5: ✅{solvable_top5} ❓{inconclusive_top5} ❌{deadlock_top5}/{len(top_boards)} | Acc Definitiva: {tpr_top5_pct:.0f}% | Tableros Únicos: {unique_boards_count}")
    return False, meta_record

def generate_final_analysis():
    print("\n📊 Agregando telemetría y generando figuras de publicación...")
    records = []
    for (lbl, heur, fo) in VARIANTS:
        for sh in SHELLS:
            for s in SEEDS:
                for c in CORES:
                    meta_path = os.path.join(OUTPUT_DIR, f"{heur}_{fo}_shell{sh}_seed{s}_cores{c}_meta.json")
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

    # Resumen por Shell, Variante y Cores
    cols_to_avg = ["Top5_Best_Real_Astar_Pushes", "Top5_Accuracy_Pct", "Time_s", "Unique_Boards_Explored", "Hybrid_Delegations_6PlusBoxes", "Deadlocks_Filtered", "Collapsed_Immediate"]
    if "Top5_Inconclusive_Count" in df.columns:
        cols_to_avg.insert(2, "Top5_Inconclusive_Count")
    summary = df.groupby(["Shell", "Variant", "Cores"])[cols_to_avg].mean().reset_index().round(1)

    if "Collapsed_Immediate" in summary.columns:
        summary["Collapse_Rate_Pct"] = (summary["Collapsed_Immediate"] * 100).round(1)
        summary.drop(columns=["Collapsed_Immediate"], inplace=True)

    print("\n" + "="*130)
    print(" 🏆 EXPERIMENTO 1: MATRIZ 2x2 COMPRENSIVA (MEDIA SOBRE 10 SEMILLAS POR CONFIGURACIÓN | 300s POR CORRIDA)")
    print("="*130)
    print(summary.to_string(index=False))

    sum_path = os.path.join(OUTPUT_DIR, "experiment_1_summary_table.csv")
    summary.to_csv(sum_path, index=False)
    print(f"📁 Tabla resumen guardada en: {sum_path}")

    # Tabla de desglose explícito de auditoría Top-5 (Total de los tableros por Shell/Variante/Cores sobre las 10 semillas)
    if "Top5_Solvable_Count" in df.columns and "Top5_Deadlock_Count" in df.columns:
        if "Top5_Inconclusive_Count" not in df.columns:
            df["Top5_Inconclusive_Count"] = 0
        
        audit_breakdown = df.groupby(["Shell", "Variant", "Cores"])[
            ["Top5_Solvable_Count", "Top5_Inconclusive_Count", "Top5_Deadlock_Count"]
        ].sum().reset_index()
        
        audit_breakdown.rename(columns={
            "Top5_Solvable_Count": "Sum_SOLVED",
            "Top5_Inconclusive_Count": "Sum_INCONCLUSIVE",
            "Top5_Deadlock_Count": "Sum_DEADLOCK_Genuino"
        }, inplace=True)
        
        # Precisión Definitiva (excluyendo Inconclusos de numerador y denominador)
        def calc_acc(row):
            def_total = row["Sum_SOLVED"] + row["Sum_DEADLOCK_Genuino"]
            return round((row["Sum_SOLVED"] / def_total) * 100.0, 1) if def_total > 0 else 0.0
            
        audit_breakdown["Definite_Accuracy_Pct (sin Inconclusos)"] = audit_breakdown.apply(calc_acc, axis=1)
        
        print("\n" + "="*130)
        print(" 🔬 DESGLOSE RIGUROSO DE AUDITORÍA TOP-5 POR SHELL Y VARIANTE (TOTAL DE TABLEROS AUDITADOS SOBRE 10 SEMILLAS)")
        print("    (Los casos INCONCLUSIVE no cuentan ni como éxito ni como fallo en 'Definite_Accuracy_Pct')")
        print("="*130)
        print(audit_breakdown.to_string(index=False))
        
        breakdown_path = os.path.join(OUTPUT_DIR, "experiment_1_audit_breakdown.csv")
        audit_breakdown.to_csv(breakdown_path, index=False)
        print(f"📁 Desglose riguroso guardado en: {breakdown_path}")

    # Gráficas
    df["Variant_Cores"] = df["Variant"] + " (" + df["Cores"].astype(str) + " cores)"
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Shell", y="Top5_Best_Real_Astar_Pushes", hue="Variant_Cores", palette="Set2")
    plt.title("Experimento 1: Calidad Certificada por A* (Pushes Reales en Top-5 Final)")
    plt.ylabel("Pushes Reales (A* Ground Truth)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp1_real_pushes_boxplot.pdf"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Shell", y="Time_s", hue="Variant_Cores", palette="mako")
    plt.title("Tiempo Computacional hasta Convergencia / Límite en Matriz 2x2")
    plt.ylabel("Tiempo Promedio (segundos)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp1_execution_time_barplot.pdf"))
    plt.close()
    print("📈 Figuras PDF guardadas en la carpeta experiment_1_matrix_results_v2/.")

def main():
    print("\n" + "="*120)
    print(" 🧪 EXPERIMENTO 1 DEFINITIVO: MATRIZ 2x2 ESTUDIO DE ABLACIÓN GENERAL (5 SHELLS x 10 SEEDS x 300s)")
    print(f" Política de Umbral por Escasez Estructural: {THRESHOLD_MAP}")
    print(f" Pipeline Hash: {CURRENT_PIPELINE_HASH}  ← Este valor debe ser idéntico en todas las corridas comparables")
    print("="*120)

    total_runs = len(VARIANTS) * len(SHELLS) * len(SEEDS) * len(CORES)
    count = 0
    
    import concurrent.futures
    for sh in SHELLS:
        th = THRESHOLD_MAP.get(sh, 0.70)
        print(f"\n⚙️  Configurando threshold del servidor Flask a {th} para Shell {sh}")
        set_server_threshold(th)
        
        # Ejecutamos de a 1 por vez (max_workers=1)
        # Esto garantiza CERO interferencia: cada experimento usa el 100% de la CPU (los 24 cores reales)
        # sin competir con otros, logrando la medición de Tiempo MÁS precisa posible.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for (lbl, heur, fo) in VARIANTS:
                for s in SEEDS:
                    for c in CORES:
                        futures.append(executor.submit(run_experiment_run, lbl, heur, fo, sh, s, c))
                        
            for f in concurrent.futures.as_completed(futures):
                count += 1
                print(f"[{count:03d}/{total_runs:03d}] Run completed.")

    set_server_threshold(0.70)
    generate_final_analysis()

if __name__ == "__main__":
    main()
