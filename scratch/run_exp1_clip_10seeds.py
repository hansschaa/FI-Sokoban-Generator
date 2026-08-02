import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "exp1_clip_10seeds_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 360
SERVER_URL = "http://127.0.0.1:5000"

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
SHELLS_TO_RUN = [2, 3]

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
            return True
    except Exception as e:
        print(f"❌ Error conectando con el servidor neuronal en {url}: {e}")
        print("💡 Asegúrate de que surrogate_server.py esté corriendo en la Terminal 1.")
        return False

def verify_board(board_flat_str, tag="clip10s"):
    if not board_flat_str or len(board_flat_str) < 5:
        return 0, "DEADLOCK"
        
    rows = [r for r in board_flat_str.split("|") if r.strip() != ""]
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{tag}.sok")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
        
    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin): return 0, "DEADLOCK"

    cmd = [solver_bin, temp_file, "0", "1000"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, text=True)
        out = proc.stdout
        pushes = 0
        status = "DEADLOCK"
        for line in out.split("\n"):
            if "Pushes:" in line:
                try: pushes = int(line.split(":")[1].strip())
                except: pass
            if "Status: SOLVED" in line: status = "SOLVED"
            elif "Status: INCONCLUSIVE" in line or "TIMEOUT" in line: status = "INCONCLUSIVE"
            elif "Status: UNSOLVABLE" in line or "DEADLOCK" in line: status = "DEADLOCK"
        if status == "DEADLOCK" and pushes > 0: status = "SOLVED"
    except subprocess.TimeoutExpired:
        pushes = 0
        status = "INCONCLUSIVE"
    except Exception:
        pushes = 0
        status = "INCONCLUSIVE"
            
    if os.path.exists(temp_file):
        try: os.remove(temp_file)
        except: pass
        
    return pushes, status

def main():
    print("\n" + "="*135)
    print(" 🌟 VALIDACIÓN RIGUROSA 10 SEMILLAS: IMPACTO DEL CLIP DE ADMISIBILIDAD EN SHELL 2 Y SHELL 3")
    print(" Verificando si el salto de calidad del Clip en Full Surrogate se sostiene en la muestra completa (Exp 1)")
    print("="*135)

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path): runner_path = "./build2/experiment_runner"
    if not os.path.exists(runner_path):
        print("❌ Error: No se encontró experiment_runner.")
        return

    exp1_table = "experiment_1_matrix_results/experiment_1_summary_table.csv"
    baseline_dict = {}
    old_full_dict = {}
    if os.path.exists(exp1_table):
        df_old = pd.read_csv(exp1_table)
        for _, row in df_old.iterrows():
            sh = int(str(row["Shell"]).replace("Shell ", "").strip())
            var = str(row["Variant"])
            pushes = float(row["Top5_Best_Real_Astar_Pushes"])
            if "A* Puro" in var or "hungarian" in var: baseline_dict[sh] = pushes
            elif "Full Surrogate" in var or "full_surrogate" in var: old_full_dict[sh] = pushes

    results_by_shell = {2: [], 3: []}

    total_tasks = len(SHELLS_TO_RUN) * len(SEEDS)
    current_task = 0

    for sh in SHELLS_TO_RUN:
        shell_file = f"levels/shell_{sh}.sok"
        if not set_server_threshold(THRESHOLD_MAP.get(sh, 0.70)):
            return

        for seed in SEEDS:
            current_task += 1
            out_csv = os.path.join(OUTPUT_DIR, f"full_surrogate_clip_shell{sh}_seed{seed}.csv")
            out_txt = os.path.join(OUTPUT_DIR, f"full_surrogate_clip_shell{sh}_seed{seed}.txt")
            meta_json = os.path.join(OUTPUT_DIR, f"full_surrogate_clip_shell{sh}_seed{seed}_meta.json")

            # Si ya fue corrido y auditado, cargar meta
            if os.path.exists(meta_json) and not "--rerun" in sys.argv:
                with open(meta_json, "r") as fm:
                    mdata = json.load(fm)
                best_real = mdata["best_real"]
                solved_cnt = mdata["solved_cnt"]
                total_boards = mdata["total_boards"]
                print(f"[{current_task}/{total_tasks}] 💡 Shell {sh} | Seed {seed:03d}: Reutilizado (Best Real: {best_real} | Acc: {solved_cnt}/{total_boards})")
            else:
                print(f"[{current_task}/{total_tasks}] ⏳ Corriendo Shell {sh} | Seed {seed:03d} (300s)...", end=" ", flush=True)
                t0 = time.time()
                try:
                    cmd = [
                        runner_path, "ES", "FO1", str(seed), shell_file,
                        "--heuristic", "full_surrogate",
                        "--timeLimit", str(TIME_LIMIT_SEC),
                        "--maxEvals", "1000000",
                        "--out_csv", out_csv
                    ]
                    with open(out_txt, "w", encoding="utf-8") as log_f:
                        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.PIPE, timeout=PYTHON_TIMEOUT)
                    dt = round(time.time() - t0, 1)
                except Exception as e:
                    print(f"❌ Falló la simulación: {e}")
                    continue

                top_boards = []
                if os.path.exists(out_txt):
                    with open(out_txt, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            if line.startswith("RANK_"):
                                parts = line.split(";")
                                if len(parts) >= 3:
                                    top_boards.append((parts[0].strip(), float(parts[1].strip()), parts[2].strip()))

                best_real = 0
                solved_cnt = 0
                for r_lbl, n_fit, b_str in top_boards:
                    real_p, st = verify_board(b_str, tag=f"sh{sh}_s{seed}_{r_lbl}")
                    if st == "SOLVED":
                        solved_cnt += 1
                        if real_p > best_real: best_real = real_p
                
                total_boards = len(top_boards)
                with open(meta_json, "w") as fm:
                    json.dump({"shell": sh, "seed": seed, "best_real": best_real, "solved_cnt": solved_cnt, "total_boards": total_boards}, fm)
                
                acc = round((solved_cnt / total_boards) * 100.0, 1) if total_boards else 0.0
                print(f"✔️ Terminado en {dt}s | Best Real: {best_real} | Precisión: {acc}% ({solved_cnt}/{total_boards})")

            results_by_shell[sh].append((seed, best_real, solved_cnt, total_boards))

    summary_rows = []
    print("\n" + "="*135)
    print(" 📊 RESULTADOS AGREGADOS (MEDIA SOBRE 10 SEMILLAS) - SHELLS 2 Y 3")
    print("="*135)

    for sh in SHELLS_TO_RUN:
        data = results_by_shell[sh]
        if not data: continue
        avg_real = round(sum(r[1] for r in data) / len(data), 1)
        tot_solved = sum(r[2] for r in data)
        tot_boards = sum(r[3] for r in data)
        acc_total = round((tot_solved / tot_boards) * 100.0, 1) if tot_boards else 0.0

        base_astar = baseline_dict.get(sh, "N/A")
        old_full = old_full_dict.get(sh, "N/A")

        summary_rows.append({
            "Shell": f"Shell {sh}",
            "A* Puro Baseline (Exp 1)": base_astar,
            "Full Surrogate SIN Clip (Media Exp 1)": old_full,
            "Full Surrogate CON Clip (Media 10 Semillas)": avg_real,
            "Precisión CON Clip": f"{acc_total}% ({tot_solved}/{tot_boards})",
            "Diferencial de Calidad con Clip": f"{avg_real - old_full:+.1f} empujes" if isinstance(old_full, float) else "N/A"
        })

    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))
    out_file = os.path.join(OUTPUT_DIR, "clip_10seeds_summary.csv")
    df_sum.to_csv(out_file, index=False)
    print(f"\n📁 Archivo de resumen guardado en: {out_file}")
    print("="*135 + "\n")

if __name__ == "__main__":
    main()
