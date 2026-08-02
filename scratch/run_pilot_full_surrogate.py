import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "pilot_full_surrogate_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 360
SERVER_URL = "http://127.0.0.1:5000"

# Política de Umbral por Shell (0.60 para Shell 3, 0.70 para el resto)
THRESHOLD_MAP = {
    1: 0.70,
    2: 0.70,
    3: 0.60,
    4: 0.70,
    5: 0.70
}

def set_server_threshold(threshold):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            return True
    except Exception as e:
        print(f"❌ Error conectando con el servidor neuronal en {url}: {e}")
        print("💡 Asegúrate de que surrogate_server.py esté corriendo.")
        return False

def verify_board_with_astar(board_flat_str, tag="temp"):
    if not board_flat_str or len(board_flat_str) < 5:
        return 0, "No Board"
        
    rows = [r for r in board_flat_str.split("|") if r.strip() != ""]
    temp_file = os.path.join(OUTPUT_DIR, f"temp_verify_{tag}.sok")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
        
    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin):
        solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin):
        return -1, "Solver Not Found"

    cmd = [solver_bin, temp_file, "0", "500"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10, text=True)
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
        
    if pushes > 1:
        return pushes, "✔️ Soluble (A* OK)"
    else:
        return 0, "❌ Deadlock (FP)"

def run_full_surrogate_pilot(shell_idx, seed=42):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = os.path.join(OUTPUT_DIR, f"ES_full_surrogate_shell{shell_idx}_seed{seed}.csv")
    out_txt = os.path.join(OUTPUT_DIR, f"ES_full_surrogate_shell{shell_idx}_seed{seed}.txt")
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

    th = THRESHOLD_MAP.get(shell_idx, 0.70)
    set_server_threshold(th)

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", "full_surrogate",
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]

    print(f"🚀 Shell {shell_idx} | Config: full_surrogate (Th={th}) | Seed {seed} (Límite: {TIME_LIMIT_SEC}s)...", end=" ", flush=True)

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

    gens = 0
    evals = 0
    deadlocks_filtered = 0
    regressor_calls = 0
    hybrid_delegations = 0
    neural_fitness = 0.0
    best_board_str_flat = ""
    top_boards = [] # list of (rank_label, neural_fit, board_str)

    lines = out_text.split('\n')
    for line in lines:
        if "[ES STATS] Classifier Deadlocks Filtered" in line:
            try: deadlocks_filtered = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Surrogate Regressor Calls" in line:
            try: regressor_calls = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Hybrid Hungarian Delegations" in line:
            try: hybrid_delegations = int(line.split(":")[1].strip())
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
                    rank_label = parts[0].strip()
                    n_fit = float(parts[1].strip())
                    b_str = parts[2].strip()
                    if n_fit > -1e8:
                        top_boards.append((rank_label, n_fit, b_str))
                except: pass
        elif line.strip() and ";" in line.strip() and not line.startswith("RANK_") and not line.startswith("["):
            parts = line.strip().split(";")
            if len(parts) >= 3:
                try: 
                    neural_fitness = -float(parts[0].strip())
                    best_board_str_flat = parts[2].strip()
                except: pass

    if neural_fitness <= -1e8 or neural_fitness >= 1e8:
        neural_fitness = 0.0

    # Si por alguna razón no se llenó top_boards, usamos el best final
    if not top_boards and best_board_str_flat:
        top_boards.append(("RANK_1 (Best)", neural_fitness, best_board_str_flat))

    print(f"✔️ Done! ({elapsed:.1f}s) | Gens: {gens} | Evals: {evals} | Filtrados: {deadlocks_filtered}")

    # Auditoría Post-Hoc A* sobre el TOP-5
    print(f"   🔬 Verificando Top-{len(top_boards)} individuos con solver A* real...")
    solvable_count = 0
    best_real_pushes = 0
    top5_records = []

    for r_label, n_fit, b_str in top_boards:
        real_p, diag = verify_board_with_astar(b_str, tag=f"sh{shell_idx}_{r_label}")
        if "Soluble" in diag:
            solvable_count += 1
            if real_p > best_real_pushes:
                best_real_pushes = real_p
        
        top5_records.append({
            "Shell": f"Shell {shell_idx}",
            "Individuo": r_label,
            "Fitness Neural": f"{n_fit:.1f}",
            "A* Real Pushes": real_p if real_p > 0 else "0 (Deadlock)",
            "Auditoría A*": diag
        })
        print(f"      • {r_label}: Neural={n_fit:.1f} | A*={real_p} -> {diag}")

    tpr_pct = (solvable_count / len(top_boards) * 100) if top_boards else 0.0
    diag_summary = f"{solvable_count}/{len(top_boards)} Solubles ({tpr_pct:.0f}%)"

    return {
        "Shell": f"Shell {shell_idx}",
        "Umbral (Th)": th,
        "Gens": gens,
        "Evals Totales": evals,
        "Filtrados (Deadlock)": deadlocks_filtered,
        "Llamadas Regresor": regressor_calls,
        "Delegaciones A* (≥6 cajas)": hybrid_delegations,
        "Top-1 Neural Fit": round(neural_fitness, 1),
        "Top-5 Mejor A*": best_real_pushes,
        "Tasa Acierto Top-5 (A*)": diag_summary,
        "Tiempo (s)": round(elapsed, 1)
    }, top5_records

def main():
    print("\n" + "="*105)
    print(" 🧪 PILOTO DE LA VARIANTE 4 (full_surrogate: CLASIFICADOR + REGRESOR SIN A* EN EL CICLO)")
    print(" Salvaguarda Metodológica: Auditoría Post-Hoc con A* Real del TOP-5 de individuos de la generación final.")
    print("="*105)

    if not set_server_threshold(0.70):
        print("\n❌ Abortando: El servidor neuronal en puerto 5000 no respondió.")
        return

    summary_results = []
    all_top5_records = []

    for shell_idx in [1, 2, 3, 4, 5]:
        res_sum, recs_top5 = run_full_surrogate_pilot(shell_idx, seed=42)
        summary_results.append(res_sum)
        all_top5_records.extend(recs_top5)

    set_server_threshold(0.70)

    print("\n" + "="*105)
    print(" 📋 RESUMEN DE COMPORTAMIENTO Y FIDELIDAD (VARIANTE 4 | SEED 42)")
    print("="*105)
    df_sum = pd.DataFrame(summary_results)
    print(df_sum.to_string(index=False))

    print("\n" + "="*105)
    print(" 🔬 DESGLOSE DE AUDITORÍA POST-HOC DEL TOP-5 POR CASCARÓN (A* REAL GROUND-TRUTH)")
    print("="*105)
    df_top5 = pd.DataFrame(all_top5_records)
    print(df_top5.to_string(index=False))

    print("\n" + "="*105)
    print(" 🏆 INTERPRETACIÓN DE LA FIDELIDAD NEURAL Y FALSOS POSITIVOS SISTEMÁTICOS")
    print("="*105)
    print("  1. Robustez del Ganador vs Población: Evaluar el Top-5 nos permite discernir si un fallo en el Top-1")
    print("     es un accidente puntual (e.g. Top-1 es FP pero Top-2/3 son excelentes solubles) o si la red se")
    print("     desorientó en una región de alta alucinación sistemática (0/5 solubles).")
    print("  2. Impacto en Velocidad: La ausencia del solver A* en el ciclo acelera las evaluaciones masivamente,")
    print("     permitiendo explorar miles de individuos en segundos y confiando en la auditoría final.")
    print("-" * 105 + "\n")

if __name__ == "__main__":
    main()
