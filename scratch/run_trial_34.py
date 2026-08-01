import sys
import optuna
sys.path.append('surrogate_models')
from optuna_path_consistency import objective

print("Starting trial 34...")
trial = optuna.trial.FixedTrial({
    "lr": 0.0000951,
    "weight_decay": 0.0001,
    "dropout_p": 0.1,
    "batch_size": 256,
    "alpha": 0.395,
    "margin": 0.179
})

acc = objective(trial)
print(f"Trial 34 returned: {acc}")
