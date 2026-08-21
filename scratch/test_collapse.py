"""
Micro-piloto de auditoría: Full Surrogate en Shell 1, semillas 43-47.
Genera un CSV por semilla con el log generación-a-generación,
y al final imprime una tabla resumen con:
  - Fitness final del regresor
  - Generaciones corridas
  - Si el disyuntor de estancamiento se disparó (fitness constante desde gen 1)
"""
import subprocess
import time
import csv
import os

SHELL = "levels/shell_1.sok"
SEEDS = [43, 44, 45, 46, 47]
RUNNER = "./build/experiment_runner"
OUT_DIR = "scratch/collapse_audit"

os.makedirs(OUT_DIR, exist_ok=True)

results = []

for seed in SEEDS:
    out_csv = os.path.join(OUT_DIR, f"full_surrogate_shell1_seed{seed}.csv")
    cmd = [
        RUNNER, "ES", "FO1", str(seed), SHELL,
        "--heuristic", "full_surrogate",
        "--timeLimit", "300",
        "--maxEvals", "1000000",
        "--out_csv", out_csv,
    ]

    print(f"\n{'='*80}")
    print(f"Seed {seed}: Lanzando Full Surrogate en Shell 1...")
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - start
    print(f"Seed {seed}: Terminó en {elapsed:.2f}s (exit code {res.returncode})")

    # Parsear el CSV generado
    if os.path.exists(out_csv):
        with open(out_csv, "r") as f:
            reader = list(csv.DictReader(f))

        num_generations = len(reader)
        if num_generations > 0:
            first_fitness = float(reader[0]["fitness"])
            last_fitness = float(reader[-1]["fitness"])
            last_evals = int(reader[-1]["evaluations"])
            last_board = reader[-1]["best_board"]

            # Detectar estancamiento: ¿el fitness fue constante desde la generación 1?
            all_same = all(float(r["fitness"]) == first_fitness for r in reader)
            stagnation_triggered = all_same

            print(f"  Generaciones: {num_generations}")
            print(f"  Fitness gen 1: {first_fitness}")
            print(f"  Fitness final: {last_fitness}")
            print(f"  Evaluaciones totales: {last_evals}")
            print(f"  Estancamiento (fitness constante): {'SÍ ⚠️' if stagnation_triggered else 'NO ✅'}")
            print(f"  CSV guardado: {out_csv}")

            results.append({
                "Seed": seed,
                "Time_s": round(elapsed, 2),
                "Generations": num_generations,
                "First_Fitness": first_fitness,
                "Final_Fitness": last_fitness,
                "Total_Evals": last_evals,
                "Stagnation": "SÍ" if stagnation_triggered else "NO",
                "Board_Identical": "SÍ" if reader[0]["best_board"] == reader[-1]["best_board"] else "NO",
            })
        else:
            print(f"  ⚠️ CSV vacío")
            results.append({"Seed": seed, "Time_s": round(elapsed, 2), "Generations": 0, "Error": "CSV vacío"})
    else:
        print(f"  ❌ No se generó CSV")
        results.append({"Seed": seed, "Time_s": round(elapsed, 2), "Generations": 0, "Error": "Sin CSV"})

# Tabla resumen final
print(f"\n{'='*80}")
print("📊 RESUMEN DE AUDITORÍA: Full Surrogate × Shell 1 × 5 Semillas")
print(f"{'='*80}")
print(f"{'Seed':<6} {'Time(s)':<9} {'Gens':<6} {'Fit_1':<8} {'Fit_Final':<10} {'Evals':<8} {'Estancado':<10} {'Board=Gen1':<10}")
print("-" * 80)
for r in results:
    if "Error" in r:
        print(f"{r['Seed']:<6} {r['Time_s']:<9} ERROR: {r['Error']}")
    else:
        print(f"{r['Seed']:<6} {r['Time_s']:<9} {r['Generations']:<6} {r['First_Fitness']:<8} {r['Final_Fitness']:<10} {r['Total_Evals']:<8} {r['Stagnation']:<10} {r['Board_Identical']:<10}")

collapse_count = sum(1 for r in results if r.get("Stagnation") == "SÍ")
print(f"\n🔬 Tasa de colapso: {collapse_count}/{len(SEEDS)}")
