import json

with open("surrogate_models/results/best_hparams.json", "r") as f:
    original = json.load(f)
    
params = original["params"]
params["alpha"] = 0.1
params["margin"] = 0.05

with open("surrogate_models/results/best_hparams_path_consistency.json", "w") as f:
    json.dump({"params": params}, f, indent=4)
