import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=12, help="Number of parallel workers")
    args = parser.parse_args()

    num_parts = args.cores
    dataset = "training_data/Solvables"
    outdir = "OriginalPathConsistency"

    # Make output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    print(f"Iniciando {num_parts} workers paralelos para minar secuencias originales de A*...")

    processes = []
    for i in range(num_parts):
        # Usamos stdout y stderr a archivos separados para no mezclar logs
        log_file = open(os.path.join(outdir, f"worker_{i}.log"), "w")
        cmd = [
            sys.executable,
            "surrogate_models/prepare_path_consistency.py",
            "--part", str(i),
            "--total-parts", str(num_parts),
            "--dataset", dataset,
            "--outdir", outdir
        ]
        
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((i, p, log_file))
        print(f"  Worker {i}/{num_parts} lanzado (PID: {p.pid})")

    print(f"\nTodos los workers corriendo. Logs en {outdir}/worker_X.log")

    # Wait for all
    for i, p, log_file in processes:
        p.wait()
        log_file.close()
        print(f"Worker {i} terminó.")

if __name__ == "__main__":
    main()
