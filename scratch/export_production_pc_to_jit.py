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

    device_cpu = torch.device("cpu")
    print(f"Exporting model on CPU...")

    # For dropout_p, we can use 0.1 as default or load from hparams.
    # The Path Consistency model used 0.1 usually.
    regressor = SokobanSEResNetRegressor(dropout_p=0.1)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No se encontró el modelo {args.model_path}")
        
    state = torch.load(args.model_path, map_location=device_cpu, weights_only=False)
    if "model_state_dict" in state:
        regressor.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        regressor.load_state_dict(state)
        
    regressor.to(device_cpu)
    regressor.eval()

    with open("surrogate_models/results/surrogate_stats.txt", "w") as sf:
        sf.write(f"3.4614\n0.8732\n")

    dummy_input_cpu = torch.randn(1, 6, 25, 25, device=device_cpu)
    traced_regressor_cpu = torch.jit.trace(regressor, dummy_input_cpu)
    
    out_jit_cpu = args.out_jit.replace(".pt", "_cpu.pt")
    traced_regressor_cpu.save(out_jit_cpu)
    print(f"¡Éxito! El modelo CPU se exportó a {out_jit_cpu}")

    if torch.cuda.is_available():
        device_cuda = torch.device("cuda")
        print(f"Exporting model on CUDA...")
        regressor.to(device_cuda)
        dummy_input_cuda = torch.randn(1, 6, 25, 25, device=device_cuda)
        traced_regressor_cuda = torch.jit.trace(regressor, dummy_input_cuda)
        out_jit_cuda = args.out_jit.replace(".pt", "_cuda.pt")
        traced_regressor_cuda.save(out_jit_cuda)
        print(f"¡Éxito! El modelo CUDA se exportó a {out_jit_cuda}")
    else:
        print("CUDA no está disponible en esta máquina, omitiendo exportación CUDA.")

if __name__ == "__main__":
    os.chdir(os.path.join(BASE_DIR, '..'))
    main()
