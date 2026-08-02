import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "pilot_shell2_thresholds_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIME_LIMIT_SEC = 300
PYTHON_TIMEOUT = 360
SERVER_URL = "http://127.0.0.1:5000"

def set_server_threshold(threshold):
    url = f"{SERVER_URL}/set_threshold"
    data = json.dumps({"threshold": threshold}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            print(f"📡 [SERVIDOR NEURAL] Umbral configurado exitosamente a: {res.get('threshold', threshold)}")
            return True
    except Exception as e:
        print(f"❌ Error al conectar con el servidor neuronal en {url}: {e}")
        print("💡 Asegúrate de que surrogate_server.py esté corriendo con el último código.")
        return False

def run_es_shell2(threshold_label, heuristic="classifier_filter"):
    shell_file = "levels/shell_2.sok"
    seed = 42
    out_csv = os.path.join(OUTPUT_DIR, f"ES_shell2_{threshold_label}.csv")
    out_txt = os.path.join(OUTPUT_DIR, f"ES_shell2_{threshold_label}.txt")
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

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", heuristic,
        "--timeLimit", str(TIME_LIMIT_SEC),
        "--maxEvals", "1000000",
        "--out_csv", tmp_csv
    ]

    print(f"🚀 Ejecutando ES en Shell 2 | Config: {threshold_label:<15} | Semilla {seed} (Límite: {TIME_LIMIT_SEC}s)...")
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, timeout=PYTHON_TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_text = result.stdout
    except subprocess.TimeoutExpired as e:
        out_text = e.stdout if e.stdout else ""
        if isinstance(out_text, bytes):
            out_text = out_text.decode('utf-8', errors='replace')
    elapsed = time.time() - start_time

    if os.path.exists(tmp_csv):
        os.rename(tmp_csv, out_csv)
    with open(out_txt, "w", encoding="utf-8", errors="replace") as f_out:
        f_out.write(out_text)

    gens = 0
    evals = 0
    deadlocks_filtered = 0
    false_positives = 0
    best_fitness = 0.0

    for line in out_text.split('\n'):
        if "[ES STATS] Classifier Deadlocks Filtered" in line:
            try: deadlocks_filtered = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Classifier False Positives" in line:
            try: false_positives = int(line.split(":")[1].strip())
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
                try: best_fitness = -float(parts[0].strip())
                except: pass

    if best_fitness == 0.0 and os.path.exists(out_csv):
        try:
            df = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df) > 0 and 'fitness' in df.columns:
                best_fitness = float(df['fitness'].iloc[-1])
                if evals == 0 and 'evaluations' in df.columns:
                    evals = int(df['evaluations'].iloc[-1])
        except: pass

    if best_fitness <= -1e8 or best_fitness >= 1e8:
        best_fitness = 0.0

    astar_evals = evals - deadlocks_filtered if heuristic == "classifier_filter" else evals
    return {
        "Umbral / Config": threshold_label,
        "Fitness Mejor": best_fitness,
        "Generaciones": gens,
        "Evals Totales": evals,
        "Deadlocks Filtrados": deadlocks_filtered,
        "Falsos Positivos (FP)": false_positives,
        "A* Reales": astar_evals,
        "Tiempo (s)": round(elapsed, 1)
    }

def main():
    print("\n" + "="*95)
    print(" 🧪 PILOTO DE RECALIBRACIÓN DE UMBRAL EN SHELL 2 (SEGUNDO REFERENTE DE ESCASEZ ESTRUCTURAL)")
    print(" Objetivo: Verificar si en Shell 2 (36% inservible) se replica el patrón de óptimo intermedio")
    print("           por debajo de 0.70 antes de fundamentar la fórmula de umbral adaptativo para el paper.")
    print("="*95)

    if not set_server_threshold(0.50):
        print("\n❌ Abortando: El servidor neuronal no respondió adecuadamente al endpoint /set_threshold.")
        return

    thresholds_to_test = [0.50, 0.55, 0.60]
    results = []

    # Correr control sin clasificador y con umbral 0.70 (o referenciar desde piloto anterior si ya se conocen)
    print("\n--- Evaluando Control Sin Clasificador (A* Puro) ---")
    res_sin = run_es_shell2("Sin Clasificador", heuristic="hungarian")
    results.append(res_sin)
    print(f"   👉 Resultado A* Puro: Fitness={res_sin['Fitness Mejor']} | Gens={res_sin['Generaciones']} | Evals={res_sin['Evals Totales']} | Tiempo={res_sin['Tiempo (s)']}s")

    print("\n--- Evaluando Control Umbral Producción (Th = 0.70) ---")
    set_server_threshold(0.70)
    res_70 = run_es_shell2("Th = 0.70 (Control)", heuristic="classifier_filter")
    results.append(res_70)
    print(f"   👉 Resultado Th=0.70: Fitness={res_70['Fitness Mejor']} | Gens={res_70['Generaciones']} | Filtrados={res_70['Deadlocks Filtrados']} | FP={res_70['Falsos Positivos (FP)']} | Tiempo={res_70['Tiempo (s)']}s")

    print("\n--- Evaluando Candidatos de Umbral Permisivo ---")
    for th in thresholds_to_test:
        print("")
        set_server_threshold(th)
        res = run_es_shell2(f"Th = {th:.2f}", heuristic="classifier_filter")
        results.append(res)
        print(f"   👉 Resultado Th={th}: Fitness={res['Fitness Mejor']} | Gens={res['Generaciones']} | Filtrados={res['Deadlocks Filtrados']} | FP={res['Falsos Positivos (FP)']} | Tiempo={res['Tiempo (s)']}s")

    # Restaurar umbral de producción
    print("\n🔄 Restaurando umbral de producción (0.70) en el servidor...")
    set_server_threshold(0.70)

    print("\n" + "="*95)
    print(" 📋 TABLA COMPARATIVA DE IMPACTO DEL UMBRAL EN SHELL 2")
    print("="*95)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\n" + "-"*95)
    print(" 🏆 DISCUSIÓN METODOLÓGICA PARA LA PROPUESTA DE UMBRAL ADAPTATIVO")
    print("  1. No-monotonía y Óptimo Intermedio: En Shell 3 se demostró que el comportamiento no es 'cuanto más")
    print("     permisivo mejor', sino que 0.70 estaba por encima de un óptimo intermedio (0.60).")
    print("  2. Segundo Punto de Muestreo (Shell 2): Si Shell 2 (también con alta tasa de esquinas/deadlocks)")
    print("     exhibe su mejor relación Exploración-Fitness en un umbral intermedio (0.55-0.60), se adquiere")
    print("     respaldo empírico para modelar T_adapt = f(C_viables / C_totales).")
    print("  3. Política para el Estudio de Ablación Completo: Para la gran comparativa general pendiente, se")
    print("     aplicará el umbral estricto (0.70) en los cascarones ricos (Shell 1, 4, 5) y el óptimo intermedio")
    print("     específico revelado para los cascarones estructuralmente escasos.")
    print("-" * 95 + "\n")

if __name__ == "__main__":
    main()
