import os
import subprocess
import time
import json
import urllib.request
import pandas as pd
import sys

OUTPUT_DIR = "pilot_shell3_thresholds_results"
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

def run_es_shell3(threshold_label, heuristic="classifier_filter"):
    shell_file = "levels/shell_3.sok"
    seed = 42
    out_csv = os.path.join(OUTPUT_DIR, f"ES_shell3_{threshold_label}.csv")
    out_txt = os.path.join(OUTPUT_DIR, f"ES_shell3_{threshold_label}.txt")
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

    print(f"🚀 Ejecutando ES en Shell 3 | Config: {threshold_label:<15} | Semilla {seed} (Límite: {TIME_LIMIT_SEC}s)...")
    
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
                try: best_fitness = float(parts[0].strip())
                except: pass

    if best_fitness == 0.0 and os.path.exists(out_csv):
        try:
            df = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df) > 0 and 'fitness' in df.columns:
                best_fitness = float(df['fitness'].iloc[-1])
                if evals == 0 and 'evaluations' in df.columns:
                    evals = int(df['evaluations'].iloc[-1])
        except: pass

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
    print(" 🧪 PILOTO DE RECALIBRACIÓN DE UMBRAL EN SHELL 3 (ESCASEZ ESTRUCTURAL)")
    print(" Objetivo: Verificar si flexibilizar el umbral del clasificador (0.50 - 0.60) rescata la")
    print("           evolución al mitigar falsos negativos en un cascarón pobre en celdas viables.")
    print("="*95)

    # Verificación inicial con el servidor
    if not set_server_threshold(0.50):
        print("\n❌ Abortando: El servidor neuronal no respondió adecuadamente al endpoint /set_threshold.")
        return

    thresholds_to_test = [0.50, 0.55, 0.60]
    results = []

    # Añadir referencias de la corrida anterior (para comparativa inmediata)
    results.append({
        "Umbral / Config": "0.70 (Control Previo)",
        "Fitness Mejor": 0.0,
        "Generaciones": 0,
        "Evals Totales": "-",
        "Deadlocks Filtrados": "-",
        "Falsos Positivos (FP)": 0,
        "A* Reales": "-",
        "Tiempo (s)": "Corto"
    })
    results.append({
        "Umbral / Config": "Sin Clasificador (A*)",
        "Fitness Mejor": 36.0,
        "Generaciones": "-",
        "Evals Totales": "-",
        "Deadlocks Filtrados": 0,
        "Falsos Positivos (FP)": 0,
        "A* Reales": "-",
        "Tiempo (s)": "Corto"
    })

    print("\n--- Evaluando Candidatos de Umbral Permisivo ---")
    for th in thresholds_to_test:
        print("")
        set_server_threshold(th)
        res = run_es_shell3(f"Th = {th:.2f}", heuristic="classifier_filter")
        results.append(res)
        print(f"   👉 Resultado Th={th}: Fitness={res['Fitness Mejor']} | Gens={res['Generaciones']} | Filtrados={res['Deadlocks Filtrados']} | Falsos Positivos={res['Falsos Positivos (FP)']} | Tiempo={res['Tiempo (s)']}s")

    # Restaurar umbral de producción
    print("\n🔄 Restaurando umbral de producción (0.70) en el servidor...")
    set_server_threshold(0.70)

    print("\n" + "="*95)
    print(" 📋 TABLA COMPARATIVA DE IMPACTO DEL UMBRAL EN SHELL 3")
    print("="*95)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\n" + "-"*95)
    print(" 🏆 ANÁLISIS METODOLÓGICO PARA EL PAPER: UMBRAL ADAPTATIVO POR ESCASEZ ESTRUCTURAL")
    print("  1. Confirmación Mecánica: Si bajar el umbral de 0.70 a 0.50-0.60 eleva el fitness de 0.0 hacia")
    print("     el nivel de referencia (~36.0), se demuestra matemáticamente la interacción entre la entropía")
    print("     topológica del cascarón y la frontera de decisión de la red neural.")
    print("  2. Asimetría del Costo de Error:")
    print("     • En Shells Ricos (e.g., Shell 1 con 83 celdas útiles), el costo del Falso Negativo es casi nulo;")
    print("       conviene un umbral estricto (0.70) para maximizar la velocidad rechazando el máximo de deadlocks.")
    print("     • En Shells Pobres (e.g., Shell 3 con 29 celdas útiles), cada Falso Negativo cercena una rama crítica")
    print("       del árbol evolutivo; conviene un umbral permisivo (0.50-0.55), tolerando un ligero incremento")
    print("       de Falsos Positivos para salvaguardar la diversidad genética.")
    print("  3. Propuesta de Umbral Adaptativo: En lugar de un parámetro estático global, se fundamenta el diseño")
    print("     de un umbral dinámico T_adapt = f(C_viables / C_totales), donde la permisividad neural escala en")
    print("     función inversa a la disponibilidad de espacio libre útil del cascarón.")
    print("-" * 95 + "\n")

if __name__ == "__main__":
    main()
