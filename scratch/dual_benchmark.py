import json
import subprocess
import os

def run_benchmark_for_lr(lr, name):
    print(f"\n{'='*55}")
    print(f"  INICIANDO PIPELINE COMPLETO PARA LR = {lr} ({name})")
    print(f"{'='*55}")
    
    with open("surrogate_models/results/best_hparams.json", "r") as f:
        base_params = json.load(f)["params"]
    
    params = base_params.copy()
    params["alpha"] = 0.1
    params["margin"] = 0.05
    params["lr"] = lr
    
    with open("surrogate_models/results/best_hparams_path_consistency.json", "w") as f:
        json.dump({"params": params}, f, indent=4)
        
    print("\n[1/4] Entrenando Fold 1 (Path Consistency)...")
    subprocess.run(["./venv/bin/python3", "surrogate_models/train_final_path_consistency.py", "--folds", "1"], check=True)
    
    print("\n[2/4] Exportando modelo a JIT (TorchScript)...")
    subprocess.run(["./venv/bin/python3", "surrogate_models/export_pc_to_jit.py", "--fold", "1"], check=True)
    
    print("\n[3/4] Ejecutando A* Benchmark (Tomará varios minutos)...")
    subprocess.run(["./venv/bin/python3", "run_benchmark.py", "--file", "sok_files/benchmark_stratified_heldout.sok", "--end", "40"], check=True)
    
    print(f"\n[4/4] Analizando intersección contra Hungarian ({name})...")
    result = subprocess.run(["./venv/bin/python3", "scratch/intersection_benchmark.py"], capture_output=True, text=True, check=True)
    
    out_file = f"scratch/intersection_output_{name}.txt"
    with open(out_file, "w") as f:
        f.write(result.stdout)
        
    print(f"-> Resumen guardado en {out_file}")
    return out_file

out_lr001 = run_benchmark_for_lr(0.001, "Mejor_Spearman")
out_lr003 = run_benchmark_for_lr(0.003, "Mejor_InterBranch")

print("\n\n" + "#"*60)
print("             VEREDICTO FINAL: EL TRADE-OFF")
print("#"*60)

print(f"\n>>> MODELO LR=0.001 (Fuerte en MAE/Spearman, débil en Inter-branch):")
with open(out_lr001, "r") as f:
    print(f.read())
    
print(f"\n>>> MODELO LR=0.003 (Débil en MAE/Spearman, fuerte en Inter-branch):")
with open(out_lr003, "r") as f:
    print(f.read())
