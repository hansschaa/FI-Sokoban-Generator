#!/usr/bin/env python3
"""
INSTRUMENTACIÓN DEFINITIVA: Semilla 44, Shell 5, Full Surrogate
================================================================
Este script corre la semilla 44 con los MISMOS parámetros del re-run v2 original,
capturando cuatro dimensiones de datos simultáneamente:

1. [C++] Timing acumulado por generación (mutación, surrogate, verificación A*)
2. [Flask] Timing server-side por batch (impreso a flask_instrumented.log)  
3. [GPU]  nvidia-smi muestreo cada 5 segundos (gpu_monitor.csv)
4. [Shell 3] Comparación cross-shell para detectar degradación sistémica
"""

import os
import subprocess
import signal
import sys
import time

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 0: Compilar (el código C++ ya viene instrumentado en el commit)
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print(" INSTRUMENTACION DEFINITIVA — Semilla 44 Shell 5 Full Surrogate")
    print("=" * 80)
    
    print("\n[PASO 0] Compilando experiment_runner...")
    r = subprocess.run(["make", "-C", "build", "-j12", "experiment_runner"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR DE COMPILACION:")
        print(r.stdout)
        print(r.stderr)
        sys.exit(1)
    print("  ✅ Compilación exitosa.")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Matar cualquier servidor Flask previo y levantar uno nuevo
    #         con logging instrumentado
    # ─────────────────────────────────────────────────────────────────────
    print("\n[PASO 1] Reiniciando servidor Flask con timing instrumentado...")
    subprocess.run("pkill -f surrogate_server.py", shell=True, capture_output=True)
    time.sleep(2)
    
    flask_log = open("scratch/flask_instrumented.log", "w")
    flask_proc = subprocess.Popen(
        ["venv/bin/python3", "surrogate_models/surrogate_server.py"],
        stdout=flask_log, stderr=flask_log
    )
    print(f"  Flask PID: {flask_proc.pid}")
    
    # Esperar a que el servidor arranque
    import urllib.request
    for attempt in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:5000/set_threshold", 
                                   data=b'{"threshold": 0.70}',
                                   timeout=3)
            print("  ✅ Flask respondiendo en puerto 5000.")
            break
        except:
            time.sleep(1)
    else:
        print("  ❌ Flask no arrancó en 60 segundos. Abortando.")
        flask_proc.kill()
        with open("scratch/flask_instrumented.log", "r") as f:
            print("=== FLASK LOG (CRASH) ===")
            print(f.read())
        sys.exit(1)
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Arrancar nvidia-smi monitor en paralelo
    # ─────────────────────────────────────────────────────────────────────
    print("\n[PASO 2] Arrancando monitor GPU (nvidia-smi cada 5s)...")
    gpu_log = open("scratch/gpu_monitor.csv", "w")
    try:
        gpu_proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu",
             "--format=csv", "-l", "5"],
            stdout=gpu_log, stderr=subprocess.DEVNULL
        )
        print(f"  nvidia-smi PID: {gpu_proc.pid}")
    except FileNotFoundError:
        print("  ⚠️  nvidia-smi no encontrado — saltando monitor GPU.")
        gpu_proc = None
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Correr semilla 44 con los parámetros EXACTOS del re-run v2
    # ─────────────────────────────────────────────────────────────────────
    # Parámetros extraídos de run_exp1_2x2_v2.py para full_surrogate:
    #   Algorithm: ES
    #   FO: FO1  
    #   --mu 10 --lambda 126 --mutRate 0.7135 --stagLimit 899
    #   --timeLimit 300 --maxEvals 1000000
    #   OMP_NUM_THREADS=1
    #   Shell file: levels/shell_5.sok
    
    runner = "./build/experiment_runner"
    if not os.path.exists(runner):
        runner = "./build2/experiment_runner"
    
    cmd_44 = [
        runner, "ES", "FO1", "44", "levels/shell_5.sok",
        "--heuristic", "full_surrogate",
        "--timeLimit", "300",
        "--maxEvals", "1000000",
        "--mu", "10", "--lambda", "126",
        "--mutRate", "0.7135", "--stagLimit", "899"
    ]
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    
    print("\n[PASO 3] Ejecutando semilla 44, Shell 5, Full Surrogate...")
    print(f"  Comando: {' '.join(cmd_44)}")
    print(f"  OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")
    print("-" * 80)
    
    t_start = time.time()
    result_44 = subprocess.run(cmd_44, env=env, capture_output=True, text=True, timeout=600)
    t_end = time.time()
    
    print("-" * 80)
    print(f"\n  ⏱️  Tiempo total Python: {t_end - t_start:.1f} s")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Correr Shell 3 para comparación cross-shell (misma semilla)
    # ─────────────────────────────────────────────────────────────────────
    print("\n[PASO 4] Ejecutando semilla 44, Shell 3, Full Surrogate (comparación)...")
    
    cmd_sh3 = [
        runner, "ES", "FO1", "44", "levels/shell_3.sok",
        "--heuristic", "full_surrogate",
        "--timeLimit", "300",
        "--maxEvals", "1000000",
        "--mu", "10", "--lambda", "126",
        "--mutRate", "0.7135", "--stagLimit", "899"
    ]
    
    t_start_sh3 = time.time()
    result_sh3 = subprocess.run(cmd_sh3, env=env, capture_output=True, text=True, timeout=600)
    t_end_sh3 = time.time()
    
    print(f"  ⏱️  Tiempo total Python (Shell 3): {t_end_sh3 - t_start_sh3:.1f} s")
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Detener procesos auxiliares y guardar logs
    # ─────────────────────────────────────────────────────────────────────
    print("\n[PASO 5] Deteniendo procesos auxiliares y guardando logs...")
    
    if gpu_proc:
        gpu_proc.terminate()
        gpu_proc.wait()
    flask_proc.terminate()
    flask_proc.wait()
    flask_log.close()
    gpu_log.close()
    
    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Imprimir TODOS los logs crudos
    # ─────────────────────────────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print(" LOG CRUDO 1: SHELL 5, SEMILLA 44 (stderr del binario C++)")
    print("=" * 80)
    for line in result_44.stderr.split('\n'):
        if any(tag in line for tag in ['[ES_TIMING]', '[PHASE_D]', '[TIMING_PHASE]', 
                                        '[ES]', 'Error:', 'ALERTA', '[ES STATS]',
                                        '[DIVERSITY]', 'CB_REJECTIONS']):
            print(line)
    
    print("\n")
    print("=" * 80)
    print(" LOG CRUDO 2: SHELL 3, SEMILLA 44 (comparación cross-shell)")
    print("=" * 80)
    for line in result_sh3.stderr.split('\n'):
        if any(tag in line for tag in ['[ES_TIMING]', '[PHASE_D]', '[ES]', 
                                        '[ES STATS]', 'CB_REJECTIONS']):
            print(line)
    
    print("\n")
    print("=" * 80)
    print(" LOG CRUDO 3: FLASK SERVER-SIDE TIMING")
    print("=" * 80)
    with open("scratch/flask_instrumented.log", "r") as f:
        for line in f:
            if "[FLASK_TIMING]" in line:
                print(line.rstrip())
    
    print("\n")
    print("=" * 80)
    print(" LOG CRUDO 4: GPU MONITOR (nvidia-smi)")  
    print("=" * 80)
    if os.path.exists("scratch/gpu_monitor.csv"):
        with open("scratch/gpu_monitor.csv", "r") as f:
            print(f.read())
    else:
        print("(no se generó)")
    
    print("\n")
    print("=" * 80)
    print(" FIN DE INSTRUMENTACIÓN — Envía todo el texto de arriba tal cual.")
    print("=" * 80)

if __name__ == "__main__":
    main()
