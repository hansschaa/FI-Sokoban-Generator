import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "pilot_full_surrogate_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 360
SERVER_URL = "http://127.0.0.1:5000"

# Política de Umbral por Shell (0.60 para Shell 3, 0.70 para el resto)
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
            res = json.loads(resp.read().decode('utf-8'))
            return True
    except Exception as e:
        print(f"❌ Error conectando con el servidor neuronal en {url}: {e}")
        print("💡 Asegúrate de que surrogate_server.py esté corriendo.")
        return False

def verify_board_with_astar(board_flat_str):
    if not board_flat_str or len(board_flat_str) < 5:
        return 0, "No Board"
        
    rows = [r for r in board_flat_str.split("|") if r.strip() != ""]
    temp_file = os.path.join(OUTPUT_DIR, "temp_verify.sok")
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
        
    solver_bin = "./build/sokoban_solver"
    if not os.path.exists(solver_bin):
        solver_bin = "./build2/sokoban_solver"
    if not os.path.exists(solver_bin):
        return -1, "Solver Not Found"

    cmd = [solver_bin, temp_file, "0", "500"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10, text=True)
    except Exception as e:
        out = getattr(e, "output", "")
        if isinstance(out, bytes): out = out.decode('utf-8', errors='replace')

    pushes = 0
    for line in out.split("\n"):
        if "Pushes:" in line:
            try: pushes = int(line.split(":")[1].strip())
            except: pass
            
    if os.path.exists(temp_file):
        try: os.remove(temp_file)
        except: pass
        
    if pushes > 1:
        return pushes, "✔️ Soluble (Certificado A*)"
    else:
        return 0, "❌ Deadlock / Hallucinado (FP)"

def run_full_surrogate_pilot(shell_idx, seed=42):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = os.path.join(OUTPUT_DIR, f"ES_full_surrogate_shell{shell_idx}_seed{seed}.csv")
    out_txt = os.path.join(OUTPUT_DIR, f"ES_full_surrogate_shell{shell_idx}_seed{seed}.txt")
    tmp_csv = out_csv + ".tmp"

    if os.path.exists(tmp_csv):
        try: os.remove(tmp_csv)
        except: pass

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path):
        runner_path = "./build2/experiment_runner"
    if not os.path.exists(runner_path):
        print("❌ Error: No se encontró experiment_runner en ./build/ ni ./build2/")
        sys.exit(1)

    th = THRESHOLD_MAP.get(shell_idx, 0.70)
    set_server_threshold(th)

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", "full_surrogate",
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]

    print(f"🚀 Shell {shell_idx} | Config: full_surrogate (Th={th}) | Seed {seed} (Límite: {TIME_LIMIT_SEC}s)...", end=" ", flush=True)

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes): out_text = out_text.decode('utf-8', errors='replace')
    elapsed = time.time() - start_time

    if os.path.exists(tmp_csv):
        os.rename(tmp_csv, out_csv)
    with open(out_txt, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write(out_text)

    gens = 0
    evals = 0
    deadlocks_filtered = 0
    regressor_calls = 0
    neural_fitness = 0.0
    board_str_flat = ""

    for line in out_text.split('\n'):
        if "[ES STATS] Classifier Deadlocks Filtered" in line:
            try: deadlocks_filtered = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Surrogate Regressor Calls" in line:
            try: regressor_calls = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Total Generations:" in line:
            parts = line.split("|")
            for p in parts:
                if "Generations:" in p:
                    try: gens = int(p.split(":")[1].strip())
                    except: pass
                elif "Evals:" in p:
                    try: evals = int(p.split(":")[1].strip())
                    except: pass
        elif line.strip() and ";" in line.strip():
            parts = line.strip().split(";")
            if len(parts) >= 3:
                try: 
                    neural_fitness = -float(parts[0].strip())
                    board_str_flat = parts[2].strip()
                except: pass

    if neural_fitness <= -1e8 or neural_fitness >= 1e8:
        neural_fitness = 0.0

    real_pushes, diagnosis = verify_board_with_astar(board_str_flat)
    
    print(f"✔️ Done! ({elapsed:.1f}s) | Gens: {gens} | Neural Fit: {neural_fitness:.1f} | A* Real: {real_pushes} ({diagnosis})")
    
    return {
        "Shell": f"Shell {shell_idx}",
        "Umbral (Th)": th,
        "Gens": gens,
        "Evals Totales": evals,
        "Filtrados (Deadlock)": deadlocks_filtered,
        "Llamados Regresor": regressor_calls,
        "Fitness Neural (Pred)": round(neural_fitness, 1),
        "A* Pushes (Real)": real_pushes,
        "Certificación A*": diagnosis,
        "Tiempo (s)": round(elapsed, 1)
    }

def main():
    print("\n" + "="*105)
    print(" 🧪 PILOTO DE LA VARIANTE 4 (full_surrogate: CLASIFICADOR + REGRESOR SIN A* EN EL CICLO)")
    print(" Objetivo: Completar la Matriz 2x2 del Experimento 1 evaluando velocidad e integridad de las soluciones")
    print("           generadas puramente mediante inferencia neuronal pre-validada por el clasificador.")
    print("="*105)

    if not set_server_threshold(0.70):
        print("\n❌ Abortando: El servidor neuronal en puerto 5000 no respondió.")
        return

    results = []
    for shell_idx in [1, 2, 3, 4, 5]:
        res = run_full_surrogate_pilot(shell_idx, seed=42)
        results.append(res)

    set_server_threshold(0.70)

    print("\n" + "="*105)
    print(" 📋 RESULTADOS DEL PILOTO VARIANTE 4 (full_surrogate | SEED 42)")
    print("="*105)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\n" + "="*105)
    print(" 🏆 SÍNTESIS DE LA MATRIZ 2x2 PARA EL EXPERIMENTO 1 DEL PAPER")
    print("="*105)
    print(" A través de estas 4 variantes aislamos de manera ortogonal el aporte de cada componente neuronal:")
    print("  • Variante 1 [A* Puro]:           Verificación de Jugabilidad = A* Real   | Decisión de Fitness = A* Real")
    print("  • Variante 2 [classifier_filter]: Verificación de Jugabilidad = Clasificador| Decisión de Fitness = A* Real")
    print("  • Variante 3 [hybrid_regressor]:  Verificación de Jugabilidad = A* Real   | Decisión de Fitness = Regresor")
    print("  • Variante 4 [full_surrogate]:    Verificación de Jugabilidad = Clasificador| Decisión de Fitness = Regresor")
    print("\n 💡 Valor Científico:")
    print("  - Comparar Var 1 vs Var 2 mide el ahorro computacional del pre-filtro contrastivo al descartar deadlocks.")
    print("  - Comparar Var 1 vs Var 3 mide el impacto en convergencia evolutiva al sustituir el árbol A* por el regresor.")
    print("  - Comparar Var 2/3 vs Var 4 mide el desempeño del pipeline 100% neuronal sin embudos de simulación simbólica.")
    print("-" * 105 + "\n")

if __name__ == "__main__":
    main()
