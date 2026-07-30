import torch
import json
import os
import argparse
from models.resnet import SokobanSEResNetRegressor

def export_fold_model(fold: int):
    device = torch.device("cpu")
    print(f"Exporting PC Fold {fold} on device: {device}")

    with open("results/best_hparams.json", "r") as f:
        r_params = json.load(f)

    regressor = SokobanSEResNetRegressor(dropout_p=r_params['params']["dropout_p"])
    
    model_path = f"results/path_consistency/final_regressor_fold{fold}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo {model_path}")
        
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        regressor.load_state_dict(state["model_state_dict"])
    else:
        regressor.load_state_dict(state)
        
    regressor.eval()

    stats_path = f"results/regressor_fold{fold}_stats.pt"
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    with open("results/surrogate_stats.txt", "w") as sf:
        sf.write(f"{stats['pushes_mean']}\n{stats['pushes_std']}\n")

    dummy_input = torch.randn(1, 6, 25, 25)
    traced_regressor = torch.jit.trace(regressor, dummy_input)
    
    out_jit = "results/surrogate_regressor_jit.pt"
    traced_regressor.save(out_jit)
    
    print(f"\n¡Éxito! El modelo Path Consistency Fold {fold} se exportó a {out_jit}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()
    
    export_fold_model(args.fold)
