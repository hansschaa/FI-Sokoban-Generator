#!/usr/bin/env python3
"""
run_experiment_rq1_rq2.py
=========================
Orquestador del experimento para RQ1 y RQ2.

USO:
    python3 scripts/run_experiment_rq1_rq2.py [--reps N] [--output archivo.csv]

ANTES DE CORRER:
    1. Esperar a que iRace termine para cada combinación Algo x FO.
    2. Rellenar el diccionario BEST_PARAMS con los hiperparámetros óptimos.
    3. Compilar: cd build && make irace_generator -j$(nproc)
"""

import subprocess
import os
import csv
import sys
import argparse
import time
from datetime import datetime

# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

BINARY      = "./build/experiment_runner"
BOARDS_FILE = "./levels/experiments_shells.txt"
OUTPUT_CSV  = "exp1_raw_data.csv"
REPS        = 30   # repeticiones por combinación (Algo x FO x Tablero)

ALGORITHMS = ["GA", "ES", "SA"]
OBJECTIVES = ["FO1", "FO4", "FO5"]

# ============================================================
# HIPERPARÁMETROS ÓPTIMOS  ← RELLENAR AL TERMINAR IRACE
# ============================================================
# Para leer resultados de iRace en R:
#   load("tuning_results/irace_GA_FO1.Rdata")
#   print(iraceResults$iterationElites)
#
# Formato: BEST_PARAMS[(Algoritmo, FO)] = { "--flag": "valor", ... }

BEST_PARAMS = {
    # ------------ GENETIC ALGORITHM ------------
    ("GA", "FO1"): {
        "--offspring": "45",
        "--maxFailed": "41",
        "--mutRate":   "0.8339",
        "--crossRate": "0.3729",
    },
    ("GA", "FO4"): {
        "--offspring": "35",
        "--maxFailed": "14",
        "--mutRate":   "0.9462",
        "--crossRate": "0.9140",
    },
    ("GA", "FO5"): {
        "--offspring": "33",
        "--maxFailed": "18",
        "--mutRate":   "0.7570",
        "--crossRate": "0.8483",
    },
    # ------------ EVOLUTION STRATEGY ------------
    ("ES", "FO1"): {
        "--mu":      "6",
        "--lambda":  "30",
        "--mutRate": "0.9878",
    },
    ("ES", "FO4"): {
        "--mu":      "15",
        "--lambda":  "20",
        "--mutRate": "0.9569",
    },
    ("ES", "FO5"): {
        "--mu":      "15",
        "--lambda":  "20",
        "--mutRate": "0.7987",
    },
    # ------------ SIMULATED ANNEALING (Re-tuned with fair stagnation limit = 600) ------------
    ("SA", "FO1"): {
        "--initTemp":  "899.4335",
        "--coolRate":  "0.0092",
        "--maxFailed": "81",
    },
    ("SA", "FO4"): {
        "--initTemp":  "665.6117",
        "--coolRate":  "0.0058",
        "--maxFailed": "51",
    },
    ("SA", "FO5"): {
        "--initTemp":  "19.3526",
        "--coolRate":  "0.0069",
        "--maxFailed": "90",
    },
}

# ============================================================
# PARSER DE TABLEROS (formato: "Shell ID: N\n...tablero...")
# ============================================================

def parse_boards(filepath):
    boards = []
    current_id    = None
    current_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")

            if line.startswith("Shell ID:"):
                if current_id is not None and current_lines:
                    # Strip leading/trailing blank lines (C++ Flood Fill is sensitive)
                    content_lines = [l for l in current_lines]
                    while content_lines and not content_lines[0].strip():
                        content_lines.pop(0)
                    while content_lines and not content_lines[-1].strip():
                        content_lines.pop()
                    boards.append({
                        "id":      current_id,
                        "content": "\n".join(content_lines)
                    })
                    current_lines = []
                current_id = int(line.split(":")[1].strip())

            elif "======" in line:
                continue

            else:
                if current_id is not None:
                    current_lines.append(line)

    if current_id is not None and current_lines:
        content_lines = [l for l in current_lines]
        while content_lines and not content_lines[0].strip():
            content_lines.pop(0)
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()
        boards.append({
            "id":      current_id,
            "content": "\n".join(content_lines)
        })

    return boards


# ============================================================
# EJECUTAR UNA COMBINACIÓN
# ============================================================

def run_one(algo, fo, seed, board_content, params, tmp_dir="./tmp_boards"):
    os.makedirs(tmp_dir, exist_ok=True)
    board_path = os.path.join(tmp_dir, f"board_{algo}_{fo}_{seed}.txt")

    with open(board_path, "w") as f:
        f.write(board_content + "\n")

    cmd = [BINARY, algo, fo, str(seed), board_path]
    for k, v in params.items():
        cmd.append(k)
        cmd.append(str(v))

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    elapsed_ms = (time.time() - t0) * 1000.0

    try:
        os.remove(board_path)
    except:
        pass

    if result.returncode != 0:
        return None, elapsed_ms, None, 0, 0

    try:
        lines = result.stdout.strip().split("\n")
        last_line = lines[-1] if lines else ""
        parts = last_line.split(";")
        if len(parts) >= 5:
            irace_cost = float(parts[0])
            board_hash = parts[1]
            evaluations = int(parts[3])
            censored = int(parts[4])
            return -irace_cost, elapsed_ms, board_hash, evaluations, censored
        elif len(parts) >= 4:
            irace_cost = float(parts[0])
            board_hash = parts[1]
            evaluations = int(parts[3])
            return -irace_cost, elapsed_ms, board_hash, evaluations, 0
        elif len(parts) >= 2:
            irace_cost = float(parts[0])
            board_hash = parts[1]
            return -irace_cost, elapsed_ms, board_hash, 0, 0   # iRace minimiza con -fitness
        else:
            # Fallback if binary didn't output hash (e.g. if we reverted)
            irace_cost = float(result.stdout.strip())
            return -irace_cost, elapsed_ms, "UNKNOWN", 0, 0
    except ValueError:
        return None, elapsed_ms, None, 0, 0


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps",   type=int, default=REPS,       help="Repeticiones por combinación")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help="Archivo CSV de salida")
    args = parser.parse_args()

    # Verificar que no queden TODOs en BEST_PARAMS
    pending = [(a, fo) for (a, fo), p in BEST_PARAMS.items()
               if any(v == "TODO" for v in p.values())]
    if pending:
        print("[ADVERTENCIA] Los siguientes parámetros aún son TODO:")
        for a, fo in pending:
            print(f"  ({a}, {fo}): {BEST_PARAMS[(a, fo)]}")
        if len(pending) == len(BEST_PARAMS):
            print("\n[ERROR] Todos los parámetros son TODO. Rellena BEST_PARAMS primero.")
            sys.exit(1)
        print("Se omitirán las combinaciones con parámetros pendientes.\n")

    if not os.path.exists(BINARY):
        print(f"[ERROR] No existe: {BINARY}")
        print("        cd build && make irace_generator -j$(nproc)")
        sys.exit(1)

    boards = parse_boards(BOARDS_FILE)
    print(f"[OK] {len(boards)} tableros cargados")

    # Count only runnable combinations (those without TODO)
    runnable = [(a, fo) for a in ALGORITHMS for fo in OBJECTIVES
                if not any(v == "TODO" for v in BEST_PARAMS.get((a, fo), {}).values())]
    total   = len(runnable) * len(boards) * args.reps
    current = 0
    errors  = 0
    start   = datetime.now()

    print(f"[OK] Combinaciones listas: {len(runnable)}/9  →  {total} ejecuciones totales")
    print(f"     {len(ALGORITHMS)} algos × {len(OBJECTIVES)} FOs × {len(boards)} tableros × {args.reps} reps")
    print(f"[->] Resultados en: {args.output}\n")

    header_needed = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if header_needed:
            writer.writerow(["Algoritmo", "FO", "Shell_ID", "Rep", "Seed",
                             "Fitness", "Elapsed_ms", "Evaluations", "Censored", "Censored_Rate", "Board_Hash", "Timestamp"])

        for algo in ALGORITHMS:
            for fo in OBJECTIVES:
                params = BEST_PARAMS.get((algo, fo), {})
                if any(v == "TODO" for v in params.values()):
                    print(f"[SKIP] ({algo}, {fo}): parámetros pendientes.")
                    continue

                for board in boards:
                    for rep in range(1, args.reps + 1):
                        current += 1
                        # Seed determinístico y sin colisiones entre combinaciones
                        base_seed = hash((algo, fo, board["id"], rep)) % (2**31)
                        seed = abs(base_seed)

                        fitness, elapsed, board_hash, evaluations, censored = run_one(algo, fo, seed,
                                                               board["content"], params)
                        if fitness is None:
                            errors += 1
                            fitness = float("nan")
                            board_hash = "ERROR"
                            evaluations = 0
                            censored = 0
                            
                        censored_rate = (censored / evaluations) if evaluations > 0 else 0.0

                        writer.writerow([
                            algo, fo, board["id"], rep, seed,
                            f"{fitness:.4f}", f"{elapsed:.1f}", evaluations, censored, f"{censored_rate:.4f}", board_hash,
                            datetime.now().strftime("%H:%M:%S")
                        ])
                        csvfile.flush()

                        elapsed_total = (datetime.now() - start).total_seconds()
                        eta = (elapsed_total / current) * (total - current) if current > 0 else 0
                        print(f"[{current:>5}/{total}] {algo}|{fo}|Shell {board['id']:>3}|"
                              f"Rep {rep:>2} | Fit={fitness:>10.3f} | ETA {eta/60:.1f}min")

    print(f"\n✅  Experimento completado. Total: {current}, Errores: {errors}")
    print(f"    Resultados: {args.output}")


if __name__ == "__main__":
    main()
