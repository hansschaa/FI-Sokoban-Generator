import json
import subprocess
import os

lrs = [0.001, 0.002, 0.003]

with open("surrogate_models/results/best_hparams.json", "r") as f:
    base_params = json.load(f)["params"]

for lr in lrs:
    print(f"\n{'='*50}")
    print(f"  TESTING LEARNING RATE: {lr}")
    print(f"{'='*50}")
    
    params = base_params.copy()
    params["alpha"] = 0.1
    params["margin"] = 0.05
    params["lr"] = lr
    
    with open("surrogate_models/results/best_hparams_path_consistency.json", "w") as f:
        json.dump({"params": params}, f, indent=4)
        
    print(f"Entrenando Fold 1 completo...")
    subprocess.run(["./venv/bin/python3", "surrogate_models/train_final_path_consistency.py", "--folds", "1"], check=False)
    
    print(f"\nEvaluando accuracy Inter-branch...")
    subprocess.run(["./venv/bin/python3", "surrogate_models/evaluate_baseline.py"], check=False)
