#!/usr/bin/env python3
"""
run_experiment_rq1_rq2.py
=========================
Orquestador del experimento para RQ1 y RQ2.

USO:
    # Correr solo un algoritmo (ideal para distribuir en 3 computadoras):
    python3 scripts/run_experiment_rq1_rq2.py --algo GA
    python3 scripts/run_experiment_rq1_rq2.py --algo ES
    python3 scripts/run_experiment_rq1_rq2.py --algo SA

    # Correr los tres algoritmos en una sola máquina:
    python3 scripts/run_experiment_rq1_rq2.py

ANTES DE CORRER:
    1. Esperar a que iRace termine para cada combinación Algo x FO.
    2. Rellenar el diccionario BEST_PARAMS con los hiperparámetros óptimos.
    3. Compilar: cd build && make experiment_runner -j$(nproc)

CRITERIO DE TÉRMINO (por corrida):
    maxEvals=1000 | stagLimit=200 — Igual para GA, ES y SA.
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
    # ------------ GENETIC ALGORITHM (Tuned with maxEvals=1000, stagLimit=200) ------------
    ("GA", "FO1"): {
        "--offspring": "29",
        "--maxFailed": "32",
        "--mutRate":   "0.8931",
        "--crossRate": "0.8913",
    },
    ("GA", "FO4"): {
        "--offspring": "29",
        "--maxFailed": "14",
        "--mutRate":   "0.9912",
        "--crossRate": "0.9491",
    },
    ("GA", "FO5"): {
        "--offspring": "39",
        "--maxFailed": "44",
        "--mutRate":   "0.8621",
        "--crossRate": "0.7030",
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
    # ------------ SIMULATED ANNEALING (Tuned with maxEvals=1000, stagLimit=200) ------------
    ("SA", "FO1"): {
        "--initTemp":  "40.208",
        "--coolRate":  "0.0238",
        "--maxFailed": "50",
    },
    ("SA", "FO4"): {
        "--initTemp":  "89.082",
        "--coolRate":  "0.0183",
        "--maxFailed": "54",
    },
    ("SA", "FO5"): {
        "--initTemp":  "676.987",
        "--coolRate":  "0.0052",
        "--maxFailed": "27",
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
    parser.add_argument("--algo",   type=str, default=None,       help="Algoritmo a ejecutar: GA, ES o SA (omitir = todos)")
    args = parser.parse_args()

    # Filtrar algoritmos según --algo y auto-nombrar el output
    algorithms_to_run = ALGORITHMS
    if args.algo:
        algo_upper = args.algo.upper()
        if algo_upper not in ALGORITHMS:
            print(f"[ERROR] --algo debe ser uno de: {', '.join(ALGORITHMS)}. Recibido: '{args.algo}'")
            sys.exit(1)
        algorithms_to_run = [algo_upper]
        # Auto-nombrar output si el usuario no especificó uno explícito
        if args.output == OUTPUT_CSV:
            args.output = f"exp1_raw_data_{algo_upper}.csv"
        print(f"[INFO] Modo single-algo: solo se ejecutará {algo_upper}")
        print(f"[INFO] Output → {args.output}")
    else:
        print(f"[INFO] Modo completo: se ejecutarán todos los algoritmos: {', '.join(ALGORITHMS)}")

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

    # Count only runnable combinations (those without TODO) for the selected algo(s)
    runnable = [(a, fo) for a in algorithms_to_run for fo in OBJECTIVES
                if not any(v == "TODO" for v in BEST_PARAMS.get((a, fo), {}).values())]
    total   = len(runnable) * len(boards) * args.reps
    current = 0
    errors  = 0
    start   = datetime.now()

    print(f"[OK] Combinaciones listas: {len(runnable)}/{len(algorithms_to_run)*len(OBJECTIVES)}  →  {total} ejecuciones totales")
    print(f"     {len(algorithms_to_run)} algo(s) × {len(OBJECTIVES)} FOs × {len(boards)} tableros × {args.reps} reps")
    print(f"[->] Resultados en: {args.output}\n")

    header_needed = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if header_needed:
            writer.writerow(["Algoritmo", "FO", "Shell_ID", "Rep", "Seed",
                             "Fitness", "Elapsed_ms", "Evaluations", "Censored", "Censored_Rate", "Board_Hash", "Timestamp"])

        for algo in algorithms_to_run:
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
