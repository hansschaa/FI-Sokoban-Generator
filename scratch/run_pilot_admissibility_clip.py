import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "pilot_admissibility_clip_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 360
SERVER_URL = "http://127.0.0.1:5000"

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

def verify_board(board_flat_str, tag="clip"):
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
    print(" 🚀 EVALUACIÓN DEL CLIP DE ADMISIBILIDAD EN FULL SURROGATE (PILOTO 5 SHELLS - SEED 42)")
    print(" Verificación en caliente: h = hungarian_lb + clamp(pred - hungarian_lb, 0, k*hungarian_lb)")
    print("="*135)

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path): runner_path = "./build2/experiment_runner"
    if not os.path.exists(runner_path):
        print("❌ Error: No se encontró experiment_runner.")
        return

    # Intentar cargar los promedios anteriores del Experimento 1 para comparativa instantánea
    exp1_table = "experiment_1_matrix_results/experiment_1_summary_table.csv"
    baseline_dict = {}
    old_full_dict = {}
    if os.path.exists(exp1_table):
        df_old = pd.read_csv(exp1_table)
        for _, row in df_old.iterrows():
            sh = int(str(row["Shell"]).replace("Shell ", "").strip())
            var = str(row["Variant"])
            pushes = row["Top5_Best_Real_Astar_Pushes"]
            if "A* Puro" in var or "hungarian" in var: baseline_dict[sh] = pushes
            elif "Full Surrogate" in var or "full_surrogate" in var: old_full_dict[sh] = pushes

    results_summary = []

    for sh in range(1, 6):
        shell_file = f"levels/shell_{sh}.sok"
        out_csv = os.path.join(OUTPUT_DIR, f"full_surrogate_clip_shell{sh}_seed42.csv")
        out_txt = os.path.join(OUTPUT_DIR, f"full_surrogate_clip_shell{sh}_seed42.txt")

        # Cargar también el resultado específico de la Semilla 42 sin clip (del Experimento 1) si existe
        old_seed42_val = "N/A"
        old_meta = f"experiment_1_matrix_results/full_surrogate_shell{sh}_seed42_meta.json"
        if os.path.exists(old_meta):
            try:
                with open(old_meta, "r") as f_meta:
                    old_seed42_val = json.load(f_meta).get("Top5_Best_Real_Astar_Pushes", "N/A")
            except: pass

        if not os.path.exists(out_txt) or "--rerun" in sys.argv:
            print(f"\n⏳ Ejecutando Full Surrogate + Clip en Shell {sh} (Semilla 42)...")
            if not set_server_threshold(THRESHOLD_MAP.get(sh, 0.70)):
                return
            cmd = [
                runner_path, "ES", "FO1", "42", shell_file,
                "--heuristic", "full_surrogate",
                "--timeLimit", str(TIME_LIMIT_SEC),
                "--maxEvals", "1000000",
                "--out_csv", out_csv
            ]
            t0 = time.time()
            try:
                with open(out_txt, "w", encoding="utf-8") as log_f:
                    proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.PIPE, timeout=PYTHON_TIMEOUT)
                dt = round(time.time() - t0, 1)
            except Exception as e:
                print(f"   ❌ Fallo o timeout en la ejecución: {e}")
                continue
        else:
            print(f"\n💡 Shell {sh}: Reutilizando simulación previa guardada en {out_txt} (usa --rerun si quieres volver a correr los 300s)...")
            dt = 0.0

        # Leer y auditar Top-5
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
            real_p, st = verify_board(b_str, tag=f"sh{sh}_{r_lbl}")
            if st == "SOLVED":
                solved_cnt += 1
                if real_p > best_real: best_real = real_p
        
        acc = round((solved_cnt / len(top_boards)) * 100.0, 1) if top_boards else 0.0
        print(f"   ✔️ Shell {sh} auditado | Precisión Top-5: {solved_cnt}/{len(top_boards)} ({acc}%) | Mejor Pushes Real con Clip: {best_real}")

        old_full = old_full_dict.get(sh, "N/A")
        base_astar = baseline_dict.get(sh, "N/A")
        
        results_summary.append({
            "Shell": f"Shell {sh}",
            "A* Puro Baseline (Media Exp 1)": base_astar,
            "Full Surrogate Sin Clip (Media Exp 1)": old_full,
            "Full Surrogate Sin Clip (Seed 42)": old_seed42_val,
            "Full Surrogate + Clip (Seed 42)": best_real,
            "Precisión con Clip": f"{acc}% ({solved_cnt}/{len(top_boards)})"
        })

    if results_summary:
        print("\n" + "="*135)
        print(" 🎯 IMPACTO DEL CLIP DE ADMISIBILIDAD SOBRE LA BRECHA DE CALIDAD")
        print("="*135)
        df_res = pd.DataFrame(results_summary)
        print(df_res.to_string(index=False))
        csv_out = os.path.join(OUTPUT_DIR, "admissibility_clip_comparison.csv")
        df_res.to_csv(csv_out, index=False)
        print(f"\n📁 Resultados exportados a: {csv_out}")
    print("="*135 + "\n")

if __name__ == "__main__":
    main()
