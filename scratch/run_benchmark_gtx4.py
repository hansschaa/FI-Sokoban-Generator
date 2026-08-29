import os
import subprocess
import time
import sys

BOARD_6 = """#################
# ##          ..#
# ##  $ $  #. # #
#      ### .$ . #
#  $# #     @. ##
###    # $ #   ##
### # ####  $ ###
##  # ## #   # ##
## #      ##    #
##### ## ### # ##
#################"""

BOARD_7 = """##############
#  ####   ####
# .# # @#$#  #
#  # .  $  $ #
##  ###$ $ #.#
##  $   #$ # #
##   .     # #
# .#.####### #
#  # #    ## #
# ## ##     .#
##############"""

def patch_and_compile():
    print("--- 1. Parcheando main_experiment.cpp para imprimir tiempos... ---")
    main_file = "src/main_experiment.cpp"
    with open(main_file, "r") as f:
        code = f.read()

    # Add chrono
    if "#include <chrono>" not in code:
        code = code.replace("#include <iostream>", "#include <iostream>\n#include <chrono>")

    # Patch start
    target1 = "Individual first_valid;\n    bool found_first = false;"
    replacement1 = "auto t_init_start = std::chrono::high_resolution_clock::now();\n    Individual first_valid;\n    bool found_first = false;"
    if "t_init_start" not in code:
        code = code.replace(target1, replacement1)

    # Patch end
    target2 = "Individual best;\n    \n    // Calcular deadlock mask"
    replacement2 = 'auto t_init_end = std::chrono::high_resolution_clock::now();\n    std::cerr << "[TIMING_INIT] Poblacion inicial generada en: " << std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count() << " ms\\n";\n    Individual best;\n    \n    // Calcular deadlock mask'
    if "t_init_end" not in code:
        code = code.replace(target2, replacement2)

    with open(main_file, "w") as f:
        f.write(code)

    print("--- Compilando binarios... ---")
    subprocess.run(["make", "-C", "build", "-j12", "experiment_runner", "batch_solver"], check=True, stdout=subprocess.DEVNULL)


def run_astar_benchmark():
    print("\n--- 2. Evaluando A* puro en tableros masivos (Shell 5) ---")
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/board_6.sok", "w") as f: f.write(BOARD_6 + "\n")
    with open("scratch/board_7.sok", "w") as f: f.write(BOARD_7 + "\n")

    env = os.environ.copy()
    env["MAX_SECONDS"] = "5.0"

    for boxes, sok_file in [(6, "scratch/board_6.sok"), (7, "scratch/board_7.sok")]:
        res_file = f"scratch/res_{boxes}.txt"
        subprocess.run(["./build/batch_solver", sok_file, "hungarian", res_file], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open(res_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                parts = lines[1].split('\t')
                print(f"[A*] Tablero de {boxes} cajas -> Status: {parts[1]} | Runtime: {parts[3]} ms | Nodos: {parts[7]}")


def run_experiment_runner():
    print("\n--- 3. Midiendo overhead de inicializacion y fallbacks en GA ---")
    print("Corriendo 100 evaluaciones de GA en shell_577.txt (semilla 44)...")
    
    cmd = [
        "./build/experiment_runner",
        "GA",
        "FO6",
        "44",
        "tuning/Instances/shell_577.txt",
        "--heuristic", "full_surrogate",
        "--maxEvals", "100",
        "--timeLimit", "300"
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "24"
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    init_time = None
    fallbacks = None
    
    for line in (res.stdout + "\n" + res.stderr).split('\n'):
        if "[TIMING_INIT]" in line:
            init_time = line.strip()
        if "Surrogate Fallbacks" in line:
            fallbacks = line.strip()
            
    if init_time:
        print(f"-> {init_time}")
    else:
        print("-> Error: No se pudo capturar el tiempo de inicializacion.")
        
    print("\n>> VERIFICACION DE FALLBACKS <<")
    # Para saber si hubo fallbacks, lo leemos del stderr si existió, pero la EA original no lo imprimía en GA.
    # Por defecto, si el server se satura por batch size, A* tomará control silencioso.
    print("Revisa el [TIMING_INIT]. Si es muy bajo (< 100 ms), NO ES EL CULPABLE.")
    print("Revisa el tiempo del tablero de 7 cajas. Si ronda los 1000+ ms, ES EL CULPABLE PRINCIPAL.")

if __name__ == "__main__":
    patch_and_compile()
    run_astar_benchmark()
    run_experiment_runner()
