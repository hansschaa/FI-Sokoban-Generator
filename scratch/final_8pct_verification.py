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
    print(" 🔬 CIERRE DEFINITIVO DEL EXPERIMENTO 1: BÚSQUEDA HISTÓRICA Y DESGLOSE SUB-POBLACIONAL (< 6 CAJAS VS >= 6 CAJAS)")
    print("="*135)

    # 1. Búsqueda profunda del tablero histórico con Neural=42.0 y examen de sus cajas
    print("\n--- 1. RASTREO PROFUNDO DEL TABLERO HISTÓRICO CON NEURAL=42.0 EN SHELL 1 ---")
    found_42 = False
    for root, dirs, files in os.walk("."):
        if ".git" in root or "build" in root or "venv" in root or ".gemini" in root: continue
        for name in files:
            if name.endswith(".txt") or name.endswith(".log") or name.endswith(".csv"):
                fp = os.path.join(root, name)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f):
                            if "42.0" in line and ("#" in line or "|" in line or "RANK_" in line or "$" in line):
                                parts = [p.strip() for p in line.split(";") if p.strip() != ""]
                                if len(parts) >= 2:
                                    # Intentar extraer tablero
                                    b_str = ""
                                    fit_str = ""
                                    for p in parts:
                                        if "#" in p and "|" in p: b_str = p
                                        elif "42" in p: fit_str = p
                                    if b_str:
                                        boxes = count_boxes(b_str)
                                        regime = "PROTEGIDO POR SWITCH (>= 6 cajas)" if boxes >= 6 else "PURAMENTE NEURAL (< 6 cajas)"
                                        print(f"📁 Encontrado en {fp} (línea {line_idx+1}) -> Fitness: {fit_str} | Cajas: {boxes} -> {regime}")
                                        print("📋 Tablero ASCII:")
                                        for r in b_str.split("|"):
                                            if r.strip(): print(f"   {r}")
                                        pushes, st = test_solver_on_board(b_str, tag="historic42")
                                        print(f"👉 Re-auditoría Corregida (Heuristic::hungarian): Pushes = {pushes} | Estado = {st}\n")
                                        found_42 = True
                except: pass
                
    if not found_42:
        print("⚠️ No se halló el archivo original con '42.0' (pudo estar en un directorio temporal limpiado por el piloto).")

    # 2. Desglose del ~8% restante en Shell 1 y 5 para Full Surrogate y las demás variantes
    print("\n" + "="*135)
    print(" 📊 DESGLOSE DE ACCURAY DE JUGABILIDAD: SUBGRUPO NEURAL PURO (< 6 CAJAS) VS SUBGRUPO PROTEGIDO A* (>= 6 CAJAS)")
    print("="*135)
    exp_dir = "experiment_1_matrix_results"
    records = []

    if os.path.exists(exp_dir):
        for heur_label, heur in [("A* Puro", "hungarian"), ("Clasificador + A*", "classifier_filter"), 
                                 ("A* verifica + Regresor", "hybrid_regressor"), ("Full Surrogate", "full_surrogate")]:
            for sh in [1, 5]:
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
                    "<6 Cajas (Neural Puro)": f"{lt6_solved}/{lt6_total} ({acc_lt6}%)",
                    "<6 Deadlocks": lt6_deadlock,
                    "<6 Inconclusos": lt6_inconclusive,
                    ">=6 Cajas (Protegido A*)": f"{ge6_solved}/{ge6_total} ({acc_ge6}%)",
                    ">=6 Deadlocks": ge6_deadlock,
                    ">=6 Inconclusos": ge6_inconclusive
                })

        if records:
            df_res = pd.DataFrame(records)
            print(df_res.to_string(index=False))
            csv_path = os.path.join(exp_dir, "subpopulation_8pct_analysis.csv")
            df_res.to_csv(csv_path, index=False)
            print(f"\n📁 Desglose sub-poblacional exportado a: {csv_path}")

    print("="*135 + "\n")

if __name__ == "__main__":
    main()
