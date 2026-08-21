import subprocess
import sys
import os

def run_pipeline(num_cores=2):
    dataset_dir = "DenseSolvables"
    out_dir = "DensePathConsistency"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: No se encontró la carpeta {dataset_dir}.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Iniciando {num_cores} workers paralelos para minar secuencias de A* (Path Consistency)...")
    
    processes = []
    
    for i in range(num_cores):
        # python3 surrogate_models/prepare_path_consistency.py --part i --total-parts N --dataset D --outdir O
        cmd = [
            "venv/bin/python3", "surrogate_models/prepare_path_consistency.py",
            "--part", str(i),
            "--total-parts", str(num_cores),
            "--dataset", dataset_dir,
            "--outdir", out_dir
        ]
        
        log_file = open(os.path.join(out_dir, f"worker_{i}.log"), "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((i, p, log_file))
        print(f"  Worker {i}/{num_cores} lanzado (PID: {p.pid})")

    print(f"\nTodos los workers corriendo. Logs en {out_dir}/worker_X.log")
    
    try:
        for i, p, log_file in processes:
            p.wait()
            log_file.close()
            print(f"Worker {i} terminó.")
    except KeyboardInterrupt:
        for i, p, log_file in processes:
            p.terminate()
            log_file.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=16)
    args = parser.parse_args()
    
    run_pipeline(args.cores)
