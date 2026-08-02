import os
import glob
import json
import subprocess
import pandas as pd

def count_boxes(board_str):
    return board_str.count('$') + board_str.count('*')

def test_solver_on_board(board_str, tag="test"):
    rows = [r for r in board_str.split("|") if r.strip() != ""]
    temp_sok = f"temp_contradiction_{tag}.sok"
    with open(temp_sok, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin): return 0, "ERROR", ""

    cmd = [solver_bin, temp_sok, "0", "1000"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        out = proc.stdout
        pushes = 0
        status = "DESCONOCIDO"
        for l in out.split("\n"):
            if "Pushes:" in l:
                try: pushes = int(l.split(":")[1].strip())
                except: pass
            if "Status:" in l:
                status = l.split(":")[1].strip()
        if status == "DESCONOCIDO" and pushes > 0: status = "SOLVED"
        elif status == "DESCONOCIDO" and pushes == 0: status = "DEADLOCK"
        if os.path.exists(temp_sok): os.remove(temp_sok)
        return pushes, status, out
    except subprocess.TimeoutExpired as e:
        if os.path.exists(temp_sok): os.remove(temp_sok)
        return 0, "INCONCLUSIVE", f"TIMEOUT > 60s: {e.stdout}"

def main():
    print("\n" + "="*130)
    print(" 🔬 RESOLUCIÓN DEFINITIVA DE CONTRADICCIÓN: ANÁLISIS DEL ROL DEL SWITCH HÍBRIDO EN FULL SURROGATE")
    print("="*130)

    # 1. Buscar cualquier archivo en pilot_full_surrogate_results o experiment_1_matrix_results de Shell 1 y 5
    print("\n--- 1. EXAMEN DE TABLEROS DE FULL SURROGATE EN SHELL 1 y SHELL 5 ---")
    for sh in [1, 5]:
        files = sorted(glob.glob(f"experiment_1_matrix_results/full_surrogate_shell{sh}_*.txt"))
        if not files:
            files = sorted(glob.glob(f"pilot_full_surrogate_results/*shell{sh}*.txt"))
        if not files:
            print(f"⚠️ No se encontraron archivos de texto para Shell {sh}.")
            continue
            
        fp = files[0]
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
            
        for line in lines:
            if line.startswith("RANK_1;") or line.startswith("RANK_"):
                parts = line.split(";")
                lbl, fit, b_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                boxes = count_boxes(b_str)
                print(f"\n📁 Muestra en Shell {sh} ({fp}) -> {lbl} | Neural Fitness: {fit} | Conteo Cajas: {boxes}")
                print("📋 Representación ASCII del Tablero:")
                for r in b_str.split("|"):
                    if r.strip(): print(f"   {r}")
                pushes, st, raw = test_solver_on_board(b_str, tag=f"sh{sh}")
                print(f"👉 Veredicto del Solver Corregido: Pushes = {pushes} | Estado = {st}")
                if "Pushes:" in raw:
                    for l in raw.split("\n"):
                        if "Pushes:" in l or "Status:" in l: print(f"   {l}")
                break

    # 2. Análisis Cuantitativo por Shell para las 4 Variantes en experiment_1_matrix_results
    print("\n" + "="*130)
    print(" 📊 TELEMETRÍA COMPRENSIVA DEL SWITCH HÍBRIDO Y DISTRIBUCIÓN DE CAJAS (MEDIA SOBRE LAS 10 SEMILLAS)")
    print("="*130)
    exp_dir = "experiment_1_matrix_results"
    records = []
    
    if os.path.exists(exp_dir):
        for heur_label, heur in [("A* Puro", "hungarian"), ("Clasificador + A*", "classifier_filter"), 
                                 ("A* verifica + Regresor", "hybrid_regressor"), ("Full Surrogate", "full_surrogate")]:
            for sh in [1, 2, 3, 4, 5]:
                meta_files = sorted(glob.glob(os.path.join(exp_dir, f"{heur}_shell{sh}_seed*_meta.json")))
                if not meta_files: continue
                
                total_evals = 0
                hybrid_dels = 0
                reg_calls = 0
                top5_total = 0
                top5_ge_6_boxes = 0
                top5_boxes_sum = 0
                
                for mf in meta_files:
                    try:
                        with open(mf, "r") as f: meta = json.load(f)
                        total_evals += meta.get("Total_Evals", 0)
                        hybrid_dels += meta.get("Hybrid_Delegations_6PlusBoxes", 0)
                        reg_calls += meta.get("Regressor_Calls", 0)
                        
                        txt_f = mf.replace("_meta.json", ".txt")
                        if os.path.exists(txt_f):
                            with open(txt_f, "r") as f_txt:
                                for line in f_txt.read().split("\n"):
                                    if line.startswith("RANK_"):
                                        parts = line.split(";")
                                        if len(parts) >= 3:
                                            top5_total += 1
                                            b_cnt = count_boxes(parts[2].strip())
                                            top5_boxes_sum += b_cnt
                                            if b_cnt >= 6: top5_ge_6_boxes += 1
                    except: pass
                
                avg_boxes = round(top5_boxes_sum / top5_total, 1) if top5_total > 0 else 0.0
                pct_ge_6 = round((top5_ge_6_boxes / top5_total) * 100.0, 1) if top5_total > 0 else 0.0
                pct_del = round((hybrid_dels / total_evals) * 100.0, 1) if total_evals > 0 else 0.0
                
                records.append({
                    "Shell": f"Shell {sh}",
                    "Variant": heur_label,
                    "Avg_Top5_Boxes": avg_boxes,
                    "Top5_Boxes_>=6_Pct": f"{pct_ge_6}%",
                    "Hybrid_Delegation_Evals_Pct": f"{pct_del}%",
                    "Avg_Hybrid_Delegations": round(hybrid_dels / len(meta_files), 0),
                    "Avg_Total_Evals": round(total_evals / len(meta_files), 0)
                })
                
        if records:
            df_res = pd.DataFrame(records)
            print(df_res.to_string(index=False))
            csv_path = os.path.join(exp_dir, "hybrid_switch_box_analysis.csv")
            df_res.to_csv(csv_path, index=False)
            print(f"\n📁 Análisis exportado a: {csv_path}")

    print("="*130 + "\n")

if __name__ == "__main__":
    main()
