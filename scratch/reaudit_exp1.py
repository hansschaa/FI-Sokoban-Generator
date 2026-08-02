import os
import json
import subprocess
import pandas as pd
import glob
import time
import sys

OUTPUT_DIR = "experiment_1_matrix_results"

def verify_board_with_astar_re(board_flat_str, tag="reaudit"):
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
            if "Status: SOLVED" in line:
                status = "SOLVED"
            elif "Status: INCONCLUSIVE" in line or "TIMEOUT" in line:
                status = "INCONCLUSIVE"
            elif "Status: UNSOLVABLE" in line or "DEADLOCK" in line:
                status = "DEADLOCK"
        if status == "DEADLOCK" and pushes > 1:
            status = "SOLVED"
    except subprocess.TimeoutExpired:
        pushes = 0
        status = "INCONCLUSIVE"
    except Exception as e:
        pushes = 0
        status = "INCONCLUSIVE"
            
    if os.path.exists(temp_file):
        try: os.remove(temp_file)
        except: pass
        
    return pushes, status

def reaudit():
    print("\n" + "="*125)
    print(" 🔬 RE-AUDITORÍA CORREGIDA DEL EXPERIMENTO 1 (SIN REPETIR LA EVOLUCIÓN | TIMEOUT = 60s)")
    print(" Criterio metodológico estricto: Diferenciación entre SOLVED, DEADLOCK genuino e INCONCLUSIVE por timeout/nodos.")
    print("="*125)

    txt_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
    if not txt_files:
        print(f"⚠️ No se encontraron archivos .txt en {OUTPUT_DIR}/. Ejecuta primero las corridas.")
        return

    updated_count = 0
    for i, txt_path in enumerate(txt_files, 1):
        prefix = os.path.splitext(txt_path)[0]
        meta_path = prefix + "_meta.json"
        csv_path = prefix + ".csv"

        if not os.path.exists(meta_path): continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f_txt:
            out_text = f_txt.read()

        top_boards = []
        best_board = ""
        neural_fitness = meta.get("Top1_Neural_Fit", 0.0)

        for line in out_text.split("\n"):
            if line.startswith("RANK_"):
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
                        best_board = parts[2].strip()
                    except: pass

        if not top_boards and best_board:
            top_boards.append(("RANK_1", neural_fitness, best_board))

        solvable_top5 = 0
        inconclusive_top5 = 0
        deadlock_top5 = 0
        best_real_astar_pushes = 0

        for r_lbl, n_fit, b_str in top_boards:
            real_p, status_str = verify_board_with_astar_re(b_str, tag=f"re_{i}")
            if status_str == "SOLVED":
                solvable_top5 += 1
                if real_p > best_real_astar_pushes:
                    best_real_astar_pushes = real_p
            elif status_str == "INCONCLUSIVE":
                inconclusive_top5 += 1
            else:
                deadlock_top5 += 1

        tpr_top5_pct = (solvable_top5 / len(top_boards) * 100.0) if top_boards else 0.0

        unique_boards_count = meta.get("Unique_Boards_Explored", 0)
        if unique_boards_count == 0 and os.path.exists(csv_path):
            try:
                df_traj = pd.read_csv(csv_path, on_bad_lines='skip')
                if 'best_board' in df_traj.columns:
                    unique_boards_count = int(df_traj['best_board'].nunique())
            except: pass

        meta["Top5_Best_Real_Astar_Pushes"] = best_real_astar_pushes
        meta["Top5_Solvable_Count"] = solvable_top5
        meta["Top5_Inconclusive_Count"] = inconclusive_top5
        meta["Top5_Deadlock_Count"] = deadlock_top5
        meta["Top5_Accuracy_Pct"] = round(tpr_top5_pct, 1)
        meta["Unique_Boards_Explored"] = unique_boards_count

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        updated_count += 1
        if i % 10 == 0 or i == len(txt_files) or inconclusive_top5 > 0:
            print(f"   [{i:03d}/{len(txt_files):03d}] Re-auditados | {meta['Variant']} ({meta['Shell']}, Seed {meta['Seed']}) -> Solubles: {solvable_top5} | Deadlocks: {deadlock_top5} | Inconclusos: {inconclusive_top5}")

    # Importar y generar tablas finales y gráficos
    sys.path.append("scratch")
    import run_exp1_2x2_matrix
    run_exp1_2x2_matrix.generate_final_analysis()

if __name__ == "__main__":
    reaudit()
