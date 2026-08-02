import os
import glob
import json
import subprocess

def count_boxes(board_str):
    return board_str.count('$') + board_str.count('*')

def test_solver_on_board(board_str, tag="test"):
    rows = [r for r in board_str.split("|") if r.strip() != ""]
    temp_sok = f"temp_contradiction_{tag}.sok"
    with open(temp_sok, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin):
        print("❌ Error: no se encuentra sokoban_solver en ./build/")
        return 0, "ERROR"

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
    print("\n" + "="*125)
    print(" 🔬 RESOLUCIÓN DEFINITIVA DE CONTRADICCIÓN: ANÁLISIS DE PILoto VIEJO Y ROL DEL SWITCH HÍBRIDO EN SHELL 1/5")
    print("="*125)

    # 1. Buscar el tablero específico del piloto con Neural=42.0 en Shell 1
    print("\n--- 1. VERIFICACIÓN DEL TABLERO HISTÓRICO DEL PILOTO (SHELL 1, NEURAL=42.0 / RANK_1) ---")
    search_dirs = ["pilot_full_surrogate_results", "experiment_1_matrix_results", "scratch", "."]
    found_historic = False
    
    for d in search_dirs:
        if not os.path.exists(d): continue
        for fp in sorted(glob.glob(os.path.join(d, "*.txt"))):
            if "shell1" in fp.lower() or "shell_1" in fp.lower():
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                for line in content.split("\n"):
                    if line.startswith("RANK_") and "; 42" in line:
                        parts = line.split(";")
                        if len(parts) >= 3:
                            lbl = parts[0].strip()
                            fit = parts[1].strip()
                            b_str = parts[2].strip()
                            boxes = count_boxes(b_str)
                            print(f"📁 Encontrado en archivo: {fp} | {lbl} | Fitness: {fit} | Cajas: {boxes}")
                            print("📋 Representación ASCII del Tablero del Piloto:")
                            for r in b_str.split("|"):
                                if r.strip(): print(f"   {r}")
                            print("\n🚀 Pasando por el solver corregido (Heuristic::hungarian):")
                            pushes, st, raw = test_solver_on_board(b_str, tag="pilot42")
                            print(raw.strip())
                            print(f"👉 Veredicto del Solver Corregido: Pushes = {pushes} | Estado = {st}")
                            found_historic = True
                            break
            if found_historic: break
        if found_historic: break

    if not found_historic:
        print("⚠️ No se encontró el archivo exacto del piloto con fitness 42.0 (¿fue sobrescrito en corridas posteriores?).")
        print("   Buscando cualquier RANK_1 en Shell 1 de Full Surrogate / neural_batched_massive para examinar:")
        for fp in sorted(glob.glob("experiment_1_matrix_results/neural_batched_massive_shell1_*.txt"))[:1]:
            with open(fp, "r") as f:
                for line in f.read().split("\n"):
                    if line.startswith("RANK_1;"):
                        parts = line.split(";")
                        lbl, fit, b_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        boxes = count_boxes(b_str)
                        print(f"📁 Archivo actual: {fp} | {lbl} | Fitness: {fit} | Cajas: {boxes}")
                        pushes, st, raw = test_solver_on_board(b_str, tag="curr_sh1")
                        print(f"👉 Veredicto del Solver Corregido: Pushes = {pushes} | Estado = {st}")
                        break

    # 2. Análisis del Rol del Switch Híbrido en Shell 1 y Shell 5 de Full Surrogate (Matriz de 200 corridas)
    print("\n--- 2. ANÁLISIS DE TELEMETRÍA Y SWITCH HÍBRIDO (BOX_COUNT >= 6) EN FULL SURROGATE (SHELL 1 y SHELL 5) ---")
    exp_dir = "experiment_1_matrix_results"
    if os.path.exists(exp_dir):
        for sh in [1, 5]:
            total_top5 = 0
            boxes_ge_6_count = 0
            total_evals_sum = 0
            hybrid_del_sum = 0
            reg_calls_sum = 0

            meta_files = sorted(glob.glob(os.path.join(exp_dir, f"neural_batched_massive_shell{sh}_seed*_meta.json")))
            for mf in meta_files:
                try:
                    with open(mf, "r") as f:
                        meta = json.load(f)
                    total_evals_sum += meta.get("Total_Evals", 0)
                    hybrid_del_sum += meta.get("Hybrid_Delegations_6PlusBoxes", 0)
                    reg_calls_sum += meta.get("Regressor_Calls", 0)

                    txt_file = mf.replace("_meta.json", ".txt")
                    if os.path.exists(txt_file):
                        with open(txt_file, "r") as f_txt:
                            for line in f_txt.read().split("\n"):
                                if line.startswith("RANK_"):
                                    parts = line.split(";")
                                    if len(parts) >= 3:
                                        total_top5 += 1
                                        if count_boxes(parts[2].strip()) >= 6:
                                            boxes_ge_6_count += 1
                except: pass

            if meta_files:
                pct_ge6 = (boxes_ge_6_count / total_top5 * 100) if total_top5 > 0 else 0
                pct_del = (hybrid_del_sum / total_evals_sum * 100) if total_evals_sum > 0 else 0
                print(f"📌 SHELL {sh} (Full Surrogate - media de {len(meta_files)} semillas):")
                print(f"   - Tableros en Top-5 con >= 6 cajas (activan switch A* real): {boxes_ge_6_count}/{total_top5} ({pct_ge6:.1f}%)")
                print(f"   - Delegaciones al Switch Híbrido A*: {hybrid_del_sum:,} / {total_evals_sum:,} evaluaciones totales ({pct_del:.1f}%)")
                print(f"   - Llamadas Puras al Regresor (< 6 cajas): {reg_calls_sum:,}")
            else:
                print(f"⚠️ No se encontraron metadatos para Shell {sh} en Full Surrogate.")

    print("="*125 + "\n")

if __name__ == "__main__":
    main()
