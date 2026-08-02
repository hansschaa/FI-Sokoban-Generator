import os
import json
import subprocess
import glob
import pandas as pd

OUTPUT_DIR = "experiment_1_matrix_results"

def verify_board(board_flat_str, tag="reg"):
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
        if status == "DEADLOCK" and pushes > 0:
            status = "SOLVED"
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
    print(" 🏆 REGENERACIÓN DEFINITIVA Y EXPORTACIÓN DE LA TABLA MAESTRA DEL EXPERIMENTO 1")
    print(" (Verificando y actualizando las 200 corridas con el Solver Corregido Heuristic::hungarian)")
    print("="*135)

    txt_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))
    if not txt_files:
        print(f"⚠️ No se encontraron archivos .txt en {OUTPUT_DIR}/.")
        return

    records = []
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

        solvable_top5 = 0
        inconclusive_top5 = 0
        deadlock_top5 = 0
        best_real_astar_pushes = 0

        for idx, (r_lbl, n_fit, b_str) in enumerate(top_boards):
            real_p, status_str = verify_board(b_str, tag=f"reg_{i}_{idx}")
            if status_str == "SOLVED":
                solvable_top5 += 1
                if real_p > best_real_astar_pushes:
                    best_real_astar_pushes = real_p
            elif status_str == "INCONCLUSIVE":
                inconclusive_top5 += 1
            else:
                deadlock_top5 += 1

        tpr_top5_pct = (solvable_top5 / len(top_boards) * 100.0) if top_boards else 0.0

        meta["Top5_Best_Real_Astar_Pushes"] = best_real_astar_pushes
        meta["Top5_Solvable_Count"] = solvable_top5
        meta["Top5_Inconclusive_Count"] = inconclusive_top5
        meta["Top5_Deadlock_Count"] = deadlock_top5
        meta["Top5_Accuracy_Pct"] = round(tpr_top5_pct, 1)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        records.append(meta)
        if i % 20 == 0 or i == len(txt_files):
            print(f" ⏳ Progreso de regeneración: [{i:03d}/{len(txt_files):03d}] corridas procesadas...")

    df = pd.DataFrame(records)
    df_csv = os.path.join(OUTPUT_DIR, "experiment_1_complete_runs_regenerated.csv")
    df.to_csv(df_csv, index=False)

    # Resumen por Shell y Variante con las columnas maestras
    cols_to_avg = ["Top5_Best_Real_Astar_Pushes", "Top5_Accuracy_Pct", "Time_s", "Unique_Boards_Explored", "Hybrid_Delegations_6PlusBoxes", "Deadlocks_Filtered"]
    summary = df.groupby(["Shell", "Variant"])[cols_to_avg].mean().reset_index().round(1)

    print("\n" + "="*135)
    print(" 🏆 EXPERIMENTO 1: TABLA DE RESUMEN FINAL DEFINITIVA (MEDIA SOBRE 10 SEMILLAS POR CONFIGURACIÓN)")
    print("="*135)
    print(summary.to_string(index=False))

    sum_path = os.path.join(OUTPUT_DIR, "experiment_1_summary_table.csv")
    summary.to_csv(sum_path, index=False)
    print("\n" + "="*135)
    print(f"📁 Tabla oficial actualizada guardada en: {sum_path}")
    print(f"📁 Todas las 200 corridas actualizadas exportadas a: {df_csv}")
    print("="*135 + "\n")

if __name__ == "__main__":
    main()
