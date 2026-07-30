import optuna
import json
import os

study = optuna.load_study(study_name='path_consistency_optuna', storage='sqlite:///path_consistency.db')
best = study.best_trial

print(f"Best Trial: {best.number} with value {best.value}")
with open("surrogate_models/results/best_hparams_path_consistency.json", "w") as f:
    json.dump(best.params, f, indent=4)
print("Saved to best_hparams_path_consistency.json")
