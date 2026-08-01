import os
import sys
import time
import hashlib
import subprocess
import requests
import json
import pandas as pd

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return "ARCHIVO NO ENCONTRADO"
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_server():
    print("="*70)
    print(" 1️⃣ VERIFICANDO MODELO DE PRODUCCIÓN Y RESPUESTA DEL SERVIDOR FLASK")
    print("="*70)
    prod_path = "surrogate_models/results/production_contrastive_classifier.pt"
    sha = compute_sha256(prod_path)
    print(f"   • Modelo de Producción : {prod_path}")
    print(f"   • SHA256 Checksum      : {sha}")
    
    parent_board = "#######\n#     #\n# @$. #\n#     #\n#######\n"
    child_board  = "#######\n#     #\n#  @* #\n#     #\n#######\n"
    payload = {"boards": [{"board": child_board, "parent_board": parent_board}]}
    
    print("   • Probando conexión con http://127.0.0.1:5000/evaluate...")
    try:
        response = requests.post("http://127.0.0.1:5000/evaluate", json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("     ✅ Servidor activo (HTTP 200 OK). Respuesta:", data)
        else:
            print(f"     ❌ Error: Servidor devolvió código HTTP {response.status_code}")
    except Exception as e:
        print(f"     ⚠️ Servidor Flask no responde en puerto 5000: {e}")
        print("        Asegúrate de que 'python surrogate_models/surrogate_server.py' esté corriendo en una terminal.")

def run_preflight_test(shell_idx, time_limit=60, seed=999):
    shell_file = f"levels/shell_{shell_idx}.sok"
    out_csv = f"scratch/preflight_shell_{shell_idx}.csv"
    if os.path.exists(out_csv):
        try: os.remove(out_csv)
        except: pass

    runner_path = "./build/experiment_runner"
    if not os.path.exists(runner_path):
        runner_path = "./build2/experiment_runner"

    cmd = [
        runner_path, "ES", "FO1", str(seed), shell_file,
        "--heuristic", "neural",
        "--timeLimit", str(time_limit),
        "--maxEvals", "100000",
        "--out_csv", out_csv
    ]
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    print(f"\n🚀 Ejecutando prueba de control en '{shell_file}' (Límite={time_limit}s, Semilla={seed})...", flush=True)
    start_t = time.time()
    try:
        res = subprocess.run(cmd, env=env, timeout=time_limit+60, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_txt = res.stdout
    except subprocess.TimeoutExpired as e:
        out_txt = e.stdout if e.stdout else ""
        if isinstance(out_txt, bytes):
            out_txt = out_txt.decode('utf-8', errors='replace')
    elapsed = time.time() - start_t

    disyuntor = 0
    deleg = 0
    gens = 0
    evals = 0
    init_att = 1
    best_fit = -1e9

    for line in out_txt.split('\n'):
        if "[ES STATS] Circuit Breaker (MAX_FAILURES) triggers:" in line:
            try: disyuntor = int(line.split(":")[1].strip())
            except: pass
        elif "[ES STATS] Hybrid Hungarian Delegations (box_count >= 6):" in line:
            try: deleg = int(line.split(":")[1].strip())
            except: pass
        elif "[INIT STATS] Initial seed found in" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "in" and i+1 < len(parts):
                    try: init_att = int(parts[i+1])
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
                try: best_fit = -float(parts[0].strip())
                except: pass

    unique_boards = 0
    if os.path.exists(out_csv):
        try:
            df = pd.read_csv(out_csv, on_bad_lines='skip')
            if len(df) > 0:
                if best_fit == -1e9 and 'fitness' in df.columns:
                    best_fit = float(df['fitness'].iloc[-1])
                if evals == 0 and 'evaluations' in df.columns:
                    evals = int(df['evaluations'].iloc[-1])
                if 'best_board' in df.columns:
                    unique_boards = len(df['best_board'].dropna().unique())
        except: pass

    print(f"   • Tiempo de reloj consumido : {elapsed:.2f}s")
    print(f"   • Intentos de Inicialización: {init_att} (Telemetría [INIT STATS])")
    print(f"   • Disparos del Disyuntor    : {disyuntor}")
    print(f"   • Delegaciones Híbridas     : {deleg}")
    print(f"   • Generaciones Evolutivas   : {gens}")
    print(f"   • Evaluaciones Totales      : {evals:,}")
    print(f"   • Fitness Mejor Individuo   : {best_fit:.1f}")
    print(f"   • Tableros Únicos Visitados : {unique_boards} (Métrica de Diversidad en CSV)")

    # Chequeo de salud metodológico
    healthy = True
    if init_att <= 0:
        print("     ⚠️ ADVERTENCIA: No se detectó telemetría de inicialización en stdout.")
    if unique_boards <= 0:
        print("     ⚠️ ADVERTENCIA: No se detectó la columna best_board en el CSV (¿Faltó re-compilar C++?).")
    if shell_idx == 5 and deleg == 0:
        print("     ⚠️ ADVERTENCIA: En Shell 5 no se registraron delegaciones híbridas (sospechoso).")
    if shell_idx == 1 and disyuntor == 0:
        print("     ℹ️ NOTA: En Shell 1 el disyuntor fue 0 (filtrado neuronal perfecto en 60s o tablero no entró en bucles de fallos).")
        
    return healthy

def main():
    verify_server()
    print("\n" + "="*70)
    print(" 2️⃣ CORRIDAS CORTAS DE PRE-VUELO SOBRE SHELL 1 Y SHELL 5 (60s C/U)")
    print("="*70)
    
    h1 = run_preflight_test(shell_idx=1, time_limit=60, seed=999)
    h5 = run_preflight_test(shell_idx=5, time_limit=60, seed=999)

    print("\n" + "="*70)
    print(" 3️⃣ EXPECTATIVA GENERAL DE TIEMPO PARA LAS 100 CORRIDAS SECUENCIALES")
    print("="*70)
    print("   • Base Teórica           : 100 corridas × 300s (5 min) = 8.33 horas de reloj.")
    print("   • Margen por Escalado    : Inicializaciones en Shell 1 y 5 pueden añadir hasta 30s por corrida.")
    print("   • Cierre de Generación   : C++ termina la generación activa antes de detener el cronómetro.")
    print("   • TIEMPO ESTIMADO TOTAL  : Entre 9.0 y 11.0 horas de ocupación continua del laboratorio.")
    print("   • Recomendación          : Ejecutar por la noche o en jornada desatendida.")
    print("="*70)

if __name__ == "__main__":
    main()
