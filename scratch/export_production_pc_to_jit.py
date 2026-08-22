import torch
import json
import os
import argparse
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'surrogate_models'))
from models.resnet import SokobanSEResNetRegressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="surrogate_models/results/path_consistency/production_path_consistency.pt")
    parser.add_argument("--out-jit", type=str, default="surrogate_models/results/production_path_consistency_jit.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting model on device: {device}")

    # For dropout_p, we can use 0.1 as default or load from hparams.
    # The Path Consistency model used 0.1 usually.
    regressor = SokobanSEResNetRegressor(dropout_p=0.1)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No se encontró el modelo {args.model_path}")
        
    state = torch.load(args.model_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        regressor.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        # Could be best_weights
        regressor.load_state_dict(state)
        
    regressor.to(device)
    regressor.eval()

    # The stats for PC are static: p_mean = 3.4614, p_std  = 0.8732
    with open("surrogate_models/results/surrogate_stats.txt", "w") as sf:
        sf.write(f"3.4614\n0.8732\n")

    dummy_input = torch.randn(1, 6, 25, 25, device=device)
    traced_regressor = torch.jit.trace(regressor, dummy_input)
    
    traced_regressor.save(args.out_jit)
    print(f"\n¡Éxito! El modelo se exportó a {args.out_jit}")

if __name__ == "__main__":
    os.chdir(os.path.join(BASE_DIR, '..'))
    main()
