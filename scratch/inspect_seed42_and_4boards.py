import os
import glob
import subprocess

def count_boxes(board_str):
    return board_str.count('$') + board_str.count('*')

def test_solver(board_str):
    rows = [r for r in board_str.split("|") if r.strip() != ""]
    temp_sok = "temp_inspect_seed42.sok"
    with open(temp_sok, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin): return "NO_SOLVER"

    cmd = [solver_bin, temp_sok, "0", "1000"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        out = proc.stdout.strip()
        if os.path.exists(temp_sok): os.remove(temp_sok)
        return out
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_sok): os.remove(temp_sok)
        return "TIMEOUT"

def main():
    print("\n" + "="*135)
    print(" 🔎 INSPECCIÓN QUIRÚRGICA: LOS 4 TABLEROS <6 CAJAS Y EL HISTORIAL DE SEED 42 EN FULL SURROGATE (SHELL 1)")
    print("="*135)
    
    exp_dir = "experiment_1_matrix_results"
    
    # 1. Examinar en detalle absoluto los 4 tableros con < 6 cajas de Shell 1 en Full Surrogate
    print("\n--- 1. RADIografía DE LOS 4 TABLEROS EXACTOS CON < 6 CAJAS EN SHELL 1 (FULL SURROGATE) ---")
    files_sh1 = sorted(glob.glob(os.path.join(exp_dir, "full_surrogate_shell1_seed*.txt")))
    count_lt6 = 0
    for fp in files_sh1:
        seed_str = fp.split("seed")[1].split("_")[0].split(".")[0]
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("RANK_"):
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) >= 3:
                        lbl, fit, b_str = parts[0], parts[1], parts[2]
                        boxes = count_boxes(b_str)
                        if boxes < 6:
                            count_lt6 += 1
                            print(f"\n🧩 Muestra #{count_lt6} | Archivo: {os.path.basename(fp)} (Semilla {seed_str}) | {lbl} | Fitness: {fit} | Cajas: {boxes}")
                            for r in b_str.split("|"):
                                if r.strip(): print(f"   {r}")
                            print(f"🚀 Resultado solver corregido:")
                            res = test_solver(b_str)
                            for l in res.split("\n"):
                                if "Pushes:" in l or "Status:" in l: print(f"   {l}")

    # 2. Inspección Específica a la Semilla 42 de Full Surrogate en Shell 1 en la matriz actual de 200 corridas
    print("\n" + "-"*135)
    print("--- 2. ANÁLISIS DE LA SEMILLA 42 EN LA MATRIZ ACTUAL (FULL SURROGATE - SHELL 1) ---")
    seed42_file = os.path.join(exp_dir, "full_surrogate_shell1_seed42.txt")
    if os.path.exists(seed42_file):
        print(f"📁 Contenido del Top-5 para Semilla 42 en el experimento de 200 corridas ({seed42_file}):")
        with open(seed42_file, "r") as f:
            for line in f:
                if line.startswith("RANK_"):
                    parts = [p.strip() for p in line.split(";")]
                    lbl, fit, b_str = parts[0], parts[1], parts[2]
                    boxes = count_boxes(b_str)
                    print(f"   {lbl} | Fitness: {fit} | Cajas: {boxes} -> ('{'PROTEGIDO SWITCH' if boxes >= 6 else 'NEURAL PURO'}')")
    else:
        print("⚠️ No se encontró el archivo de la Semilla 42 en experiment_1_matrix_results.")

    # 3. Conclusión explícita sobre la contradicción y el tablero del piloto anterior
    print("\n" + "="*135)
    print(" 💡 DIAGNÓSTICO DEFINITIVO SOBRE EL TABLERO 'NEURAL=42.0' DEL PILOTO ANTERIOR")
    print("="*135)
    print(" 👉 Con esta inspección verificamos de forma inequívoca si el tablero Neural=42.0 del piloto preliminar")
    print("    de 1 sola semilla apareció o no en esta tanda oficial de 200 corridas (10 semillas).")
    print("    Si no coincide con los 4 tableros presentados arriba, se confirma que el piloto anterior fue un experimento")
    print("    independiente cuyo resultado no entró al Top-5 de las 200 corridas actuales, disipando cualquier duda de bug.")
    print("="*135 + "\n")

if __name__ == "__main__":
    main()
