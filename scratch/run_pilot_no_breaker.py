#!/usr/bin/env python3
import os
import subprocess
import sys
import time

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    runner = "./build/experiment_runner"
    
    print("=" * 80)
    print(" VERIFICACION DE PILOTO: Sin Circuit-Breaker ni Diagnóstico")
    print("=" * 80)
    
    # Arrancar Flask
    print("\n[PASO 1] Levantando servidor Flask...")
    subprocess.run("pkill -f surrogate_server.py", shell=True, capture_output=True)
    time.sleep(2)
    flask_log = open("scratch/flask_pilot.log", "w")
    flask_proc = subprocess.Popen(
        ["venv/bin/python3", "surrogate_models/surrogate_server.py"],
        stdout=flask_log, stderr=flask_log
    )
    import urllib.request
    for attempt in range(60):
        try:
            req = urllib.request.Request("http://127.0.0.1:5000/set_threshold", data=b'{"threshold": 0.70}')
            req.add_header('Content-Type', 'application/json')
            urllib.request.urlopen(req, timeout=3)
            break
        except Exception:
            time.sleep(1)
    else:
        print("  ❌ Flask no arrancó. Abortando.")
        sys.exit(1)
    print("  ✅ Servidor listo.")

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'

    # Configuración base de parámetros reales
    base_cmd = [
        runner, "ES", "FO1", "PLACEHOLDER_SEED", "levels/shell_5.sok",
        "--timeLimit", "300", "--maxEvals", "1000000",
        "--mu", "10", "--lambda", "126",
        "--mutRate", "0.7135", "--stagLimit", "899",
        "--heuristic", "PLACEHOLDER_HEURISTIC"
    ]

    tests = [
        ("full_surrogate", "44"),
        ("full_surrogate", "45"),
        ("hybrid_regressor", "44"),
        ("hybrid_regressor", "45")
    ]

    for heuristic, seed in tests:
        cmd = list(base_cmd)
        cmd[3] = seed
        cmd[-1] = heuristic
        
        print(f"\n▶ Ejecutando {heuristic} (Semilla {seed}, Shell 5)...")
        t_start = time.time()
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=310)
        t_end = time.time()
        
        # Parse final lines
        stderr_lines = res.stderr.split('\n')
        
        total_evals = 0
        total_gens = 0
        final_best = 0.0
        cb_rejections = 0
        
        for line in stderr_lines:
            if "Total Evals" in line:
                # [ES STATS] Total Generations: X | Total Evals: Y
                parts = line.split('|')
                total_gens = int(parts[0].split(':')[-1].strip())
                total_evals = int(parts[1].split(':')[-1].strip())
            if "CB_REJECTIONS=" in line:
                cb_rejections = int(line.split('=')[-1].strip())
        
        # Find best fitness from last generation log
        for line in reversed(stderr_lines):
            if "[ES_TIMING] Gen" in line or "[ES] Gen" in line:
                parts = line.split('|')
                for p in parts:
                    if "Best Fit:" in p or "Best:" in p:
                        final_best = float(p.split(':')[-1].strip())
                        break
                break

        print(f"  ⏱️  Tiempo: {t_end - t_start:.2f}s")
        print(f"  📊  Generaciones: {total_gens} | Evals: {total_evals} | Best Fit: {final_best}")
        print(f"  ⛔  CB Rejections: {cb_rejections} (debería ser el número de rechazos por mutación fallida/timeout, no por A* elitista)")
        
        # Check if any [PHASE_D] prints leaked
        phase_d_count = sum(1 for x in stderr_lines if "[PHASE_D]" in x)
        if phase_d_count > 0:
            print(f"  ❌ ALERTA: Se imprimieron {phase_d_count} líneas [PHASE_D]. El Circuit Breaker sigue filtrando.")
        else:
            print(f"  ✅ No hay llamadas silenciosas de A* (0 líneas PHASE_D).")

if __name__ == "__main__":
    main()
