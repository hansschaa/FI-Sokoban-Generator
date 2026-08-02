import os
import glob
import subprocess
import json

OUTPUT_DIR = "experiment_1_matrix_results"

def main():
    print("\n" + "="*110)
    print(" 🧪 PRUEBA DE AUDITORÍA INDIVIDUAL EN UN TABLERO DEL TOP-5 DE SHELL 1 (A* PURO / HUNGARIAN)")
    print(" Objetivo: Verificar con el solver corregido (Heuristic::hungarian) un tablero que dio 0 pushes.")
    print("="*110)

    # Buscar una corrida de hungarian en Shell 1
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "hungarian_shell1_*.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*shell1_*.txt")))
    if not files:
        print(f"❌ No se encontraron archivos de Shell 1 en {OUTPUT_DIR}/. Asegúrate de estar en el directorio correcto.")
        return

    target_file = files[0]
    meta_file = os.path.splitext(target_file)[0] + "_meta.json"
    print(f"📁 Analizando archivo: {target_file}")

    with open(target_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")

    top_boards = []
    for line in lines:
        if line.startswith("RANK_"):
            parts = line.split(";")
            if len(parts) >= 3:
                r_lbl = parts[0].strip()
                try: n_fit = float(parts[1].strip())
                except: n_fit = 0.0
                b_str = parts[2].strip()
                top_boards.append((r_lbl, n_fit, b_str))

    if not top_boards:
        print("❌ No se encontraron líneas RANK_ en el archivo.")
        return

    # Tomar el primer tablero (RANK_1)
    r_lbl, fit, b_str = top_boards[0]
    print(f"\n👉 Seleccionado: {r_lbl} | Fitness asignada durante la evolución: {fit}")
    print("📋 Representación del Tablero:")
    rows = [r for r in b_str.split("|") if r.strip() != ""]
    for row in rows:
        print(f"   {row}")

    temp_sok = "temp_test_shell1.sok"
    with open(temp_sok, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin): solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin):
        print("❌ Error: no se encuentra el binario sokoban_solver en ./build/ ni ./build2/")
        return

    cmd = [solver_bin, temp_sok, "0", "1000"]
    print(f"\n🚀 Ejecutando solver corregido: {' '.join(cmd)}")
    print("-" * 110)

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=65)
        print(proc.stdout)
        if proc.stderr:
            print("[STDERR]:", proc.stderr)
        
        pushes = 0
        status = "DESCONOCIDO"
        for l in proc.stdout.split("\n"):
            if "Pushes:" in l:
                try: pushes = int(l.split(":")[1].strip())
                except: pass
            if "Status:" in l:
                status = l.split(":")[1].strip()

        if status == "DESCONOCIDO" and pushes > 0:
            status = "SOLVED (confirmado por Pushes > 0)"
        elif status == "DESCONOCIDO" and pushes == 0:
            status = "DEADLOCK / IRRESOLUBLE"

        print("-" * 110)
        print(f"🎯 Resultado de Re-auditoría Corregida: Pushes = {pushes} | Estado = {status}")
        print(f"⚖️ Coherencia con Fitness Evolutivo: Todo en orden si Pushes > 0 y Estado == SOLVED.")

    except subprocess.TimeoutExpired as e:
        print("-" * 110)
        print("⏳ TIMEOUT EXCEDIDO (>65s). El tablero es INCONCLUSO por límite de tiempo de simulación exhaustiva A*.")
        print(f"   Salida capturada antes de timeout: {e.stdout}")

    if os.path.exists(temp_sok):
        try: os.remove(temp_sok)
        except: pass

    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
