import subprocess
import sys
import os
import time

def run_miners(num_cores=16, pairs_per_core=5000):
    input_dir = "DenseSolvables"
    base_out_dir = "DensePairsWorkers"
    
    if not os.path.exists(input_dir):
        print(f"Error: No se encontró la carpeta {input_dir}. Asegurate de estar en la raíz del proyecto.")
        return

    os.makedirs(base_out_dir, exist_ok=True)
    
    print(f"Iniciando {num_cores} mineros de pares en paralelo...")
    print(f"Cada uno generará {pairs_per_core} pares (Total esperado: {num_cores * pairs_per_core})")
    
    processes = []
    
    for i in range(1, num_cores + 1):
        core_out = os.path.join(base_out_dir, f"worker_{i}")
        os.makedirs(core_out, exist_ok=True)
        
        # ./build/solvable_pair_miner <solvables_dir> <output_dir> <num_pairs>
        cmd = ["./build/solvable_pair_miner", input_dir, core_out, str(pairs_per_core)]
        
        # Redirigir stdout y stderr a un log para no inundar la consola
        log_file = open(os.path.join(core_out, "miner.log"), "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((i, p, log_file))
        print(f"  Worker {i} lanzado (PID: {p.pid}) -> Guardando en {core_out}")

    print("\nTodos los workers están corriendo. Podés monitorear los logs en DensePairsWorkers/worker_X/miner.log")
    print("Esperando a que terminen... (presioná Ctrl+C para matar todo)")
    
    try:
        for i, p, log_file in processes:
            p.wait()
            log_file.close()
            print(f"Worker {i} terminó.")
    except KeyboardInterrupt:
        print("\nInterrupción detectada. Matando workers...")
        for i, p, log_file in processes:
            p.terminate()
            log_file.close()
        print("Workers terminados.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=16, help="Número de workers en paralelo")
    parser.add_argument("--pairs", type=int, default=5000, help="Pares a minar por worker")
    args = parser.parse_args()
    
    run_miners(args.cores, args.pairs)
