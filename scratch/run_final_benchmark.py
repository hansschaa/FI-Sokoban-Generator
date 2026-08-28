import subprocess
import os
import pandas as pd

def main():
    sok_file = 'sok_files/benchmark_stratified_heldout.sok'
    out_csv = "surrogate_models/results/final_benchmark_results.csv"
    
    with open(out_csv, "w") as f:
        f.write("LevelName,Heuristic,Device,Status,Runtime_ms,Pushes,ExpandedNodes,TotalChildren,EffectiveChildren\n")
        
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    
    configs = [
        ("neural_sequential", "GPU", {}),
        ("neural_sequential", "CPU", {"USE_CPU": "1"}),
        ("neural_batched", "GPU", {}),
        ("neural_batched", "CPU", {"USE_CPU": "1"}),
        ("neural_batched_massive", "GPU", {}),
        ("neural_batched_massive", "CPU", {"USE_CPU": "1"})
    ]
    
    for h, device, extra_env in configs:
        print(f"\n--- Evaluando {h} en {device} ---")
        curr_env = env.copy()
        curr_env.update(extra_env)
        
        temp_out = "temp_out.tsv"
        cmd = ["./build/batch_solver", sok_file, h, temp_out]
        
        try:
            subprocess.run(cmd, env=curr_env, check=True)
            
            with open(temp_out, "r") as f:
                lines = f.read().strip().split('\n')[1:] # skip header
                
            with open(out_csv, "a") as f:
                for line in lines:
                    if not line.strip(): continue
                    parts = line.split('\t')
                    lvl = parts[0]
                    status = parts[1]
                    runtime = parts[3]
                    pushes = parts[4]
                    nodes = parts[7]
                    children = parts[8]
                    eff_children = parts[9]
                    
                    f.write(f"{lvl},{h},{device},{status},{runtime},{pushes},{nodes},{children},{eff_children}\n")
            os.remove(temp_out)
        except Exception as e:
            print(f"Error evaluando {h} en {device}: {e}")

    df = pd.read_csv(out_csv)
    print("\n=== TABLA DE RESULTADOS FINALES ===")
    print(df.to_string())

if __name__ == '__main__':
    main()
