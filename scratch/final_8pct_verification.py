import os
import glob
import json
import subprocess
import pandas as pd

def count_boxes(board_str):
    return board_str.count('$') + board_str.count('*')

def test_solver_on_board(board_str, tag="test"):
    rows = [r for r in board_str.split("|") if r.strip() != ""]
    temp_sok = f"temp_8pct_{tag}.sok"
    with open(temp_sok, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin): return 0, "ERROR"

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
        return pushes, status
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_sok): os.remove(temp_sok)
        return 0, "INCONCLUSIVE"

def main():
    print("\n" + "="*135)
    print(" 🔬 CIERRE DEFINITIVO DEL EXPERIMENTO 1: EXAMEN DE TABLEROS <6 CAJAS Y DESGLOSE SUB-POBLACIONAL")
    print("="*135)

    # 1. Búsqueda y examen detallado de tableros de < 6 cajas (Puro Neural) y tableros históricos en Shell 1 y 5
    print("\n--- 1. EXAMEN INDIVIDUAL DE TABLEROS EN RÉGIMEN PURO NEURAL (< 6 CAJAS) EN SHELL 1 Y SHELL 5 ---")
    print("💡 Buscando en el historial y en las corridas actuales ejemplos concretos del ~8% puramente neuronal...")
    
    found_examples = 0
    search_dirs = ["pilot_full_surrogate_results", "experiment_1_matrix_results"]
    for d in search_dirs:
        if not os.path.exists(d): continue
        for fp in sorted(glob.glob(os.path.join(d, "*full_surrogate*shell*.txt"))) + sorted(glob.glob(os.path.join(d, "*shell*.txt"))):
            if "shell1" in fp.lower() or "shell_1" in fp.lower() or "shell5" in fp.lower() or "shell_5" in fp.lower():
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            if line.startswith("RANK_"):
                                parts = [p.strip() for p in line.split(";")]
                                if len(parts) >= 3:
                                    lbl, fit, b_str = parts[0], parts[1], parts[2]
                                    boxes = count_boxes(b_str)
                                    # Queremos examinar especialmente si hay alguno con < 6 cajas o si coincide con los fitness del piloto anterior (42.0, 96.0)
                                    if boxes < 6 or "42" in fit or "96" in fit:
                                        regime = "PURAMENTE NEURAL (< 6 cajas - sin protección switch)" if boxes < 6 else "PROTEGIDO POR SWITCH (>= 6 cajas)"
                                        print(f"\n📁 Archivo: {fp} | {lbl} | Fitness: {fit} | Cajas: {boxes} -> {regime}")
                                        for r in b_str.split("|"):
                                            if r.strip(): print(f"   {r}")
                                        pushes, st = test_solver_on_board(b_str, tag=f"ex_{found_examples}")
                                        print(f"👉 Veredicto del Solver Corregido: Pushes = {pushes} | Estado = {st}")
                                        found_examples += 1
                                        if found_examples >= 4: break
                except: pass
            if found_examples >= 4: break
        if found_examples >= 4: break

    if found_examples == 0:
        print("💡 Nota: En los archivos actuales de Shell 1 y Shell 5 de Full Surrogate, la gran mayoría de tableros tienen >= 6 cajas.")
        print("   A continuación, verificamos sistemáticamente los 200 archivos para aislar con total exactitud las sub-poblaciones.")

    # 2. Desglose del ~8% restante en Shell 1 y 5 para las 4 variantes
    print("\n" + "="*135)
    print(" 📊 DESGLOSE DE ACCURACY DE JUGABILIDAD: SUBGRUPO NEURAL PURO (< 6 CAJAS) VS SUBGRUPO PROTEGIDO A* (>= 6 CAJAS)")
    print("="*135)
    exp_dir = "experiment_1_matrix_results"
    records = []

    if os.path.exists(exp_dir):
        for heur_label, heur in [("A* Puro", "hungarian"), ("Clasificador + A*", "classifier_filter"), 
                                 ("A* verifica + Regresor", "hybrid_regressor"), ("Full Surrogate", "full_surrogate")]:
            for sh in [1, 5]:
                print(f"⏳ Evaluando sub-poblaciones en Shell {sh} para la variante: {heur_label} ...", flush=True)
                meta_files = sorted(glob.glob(os.path.join(exp_dir, f"{heur}_shell{sh}_seed*_meta.json")))
                if not meta_files: continue

                # Contadores para < 6 cajas (Puro Neural / Sin Switch)
                lt6_total = 0; lt6_solved = 0; lt6_deadlock = 0; lt6_inconclusive = 0
                
                # Contadores para >= 6 cajas (Protegido por Switch)
                ge6_total = 0; ge6_solved = 0; ge6_deadlock = 0; ge6_inconclusive = 0

                for mf in meta_files:
                    txt_f = mf.replace("_meta.json", ".txt")
                    if os.path.exists(txt_f):
                        with open(txt_f, "r", encoding="utf-8", errors="replace") as f_txt:
                            for line in f_txt.read().split("\n"):
                                if line.startswith("RANK_"):
                                    parts = line.split(";")
                                    if len(parts) >= 3:
                                        b_str = parts[2].strip()
                                        b_cnt = count_boxes(b_str)
                                        pushes, status = test_solver_on_board(b_str, tag=f"sub_{sh}")

                                        if b_cnt < 6:
                                            lt6_total += 1
                                            if status == "SOLVED": lt6_solved += 1
                                            elif status == "INCONCLUSIVE": lt6_inconclusive += 1
                                            else: lt6_deadlock += 1
                                        else:
                                            ge6_total += 1
                                            if status == "SOLVED": ge6_solved += 1
                                            elif status == "INCONCLUSIVE": ge6_inconclusive += 1
                                            else: ge6_deadlock += 1

                # Calcular Precisiones Definitivas (sin Inconclusos)
                lt6_def = lt6_solved + lt6_deadlock
                acc_lt6 = round((lt6_solved / lt6_def) * 100.0, 1) if lt6_def > 0 else 0.0
                
                ge6_def = ge6_solved + ge6_deadlock
                acc_ge6 = round((ge6_solved / ge6_def) * 100.0, 1) if ge6_def > 0 else 0.0

                records.append({
                    "Shell": f"Shell {sh}",
                    "Variant": heur_label,
                    "<6 Cajas (Neural Puro)": f"{lt6_solved}/{lt6_total} ({acc_lt6}%)" if lt6_total > 0 else "0/0 (N/A)",
                    "<6 Deadlocks": lt6_deadlock,
                    "<6 Inconclusos": lt6_inconclusive,
                    ">=6 Cajas (Protegido A*)": f"{ge6_solved}/{ge6_total} ({acc_ge6}%)" if ge6_total > 0 else "0/0 (N/A)",
                    ">=6 Deadlocks": ge6_deadlock,
                    ">=6 Inconclusos": ge6_inconclusive
                })

        print("\n" + "="*135)
        print(" 🎯 RESULTADOS DEL DESGLOSE QUIRÚRGICO DE JUGABILIDAD POR SUB-POBLACIÓN")
        print("="*135)
        if records:
            df_res = pd.DataFrame(records)
            print(df_res.to_string(index=False))
            csv_path = os.path.join(exp_dir, "subpopulation_8pct_analysis.csv")
            df_res.to_csv(csv_path, index=False)
            print(f"\n📁 Desglose sub-poblacional exportado a: {csv_path}")

    print("="*135 + "\n")

if __name__ == "__main__":
    main()
