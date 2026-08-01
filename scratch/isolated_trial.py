import sys
import optuna
sys.path.append('surrogate_models')
from optuna_path_consistency import objective

print("Starting isolated trial...")
try:
    trial = optuna.trial.FixedTrial({
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "dropout_p": 0.2,
        "batch_size": 256,
        "alpha": 0.1,
        "margin": 0.1
    })
    
    val = objective(trial)
    print(f"Isolated trial finished. Return value: {val}")
except Exception as e:
    print(f"Error: {e}")
