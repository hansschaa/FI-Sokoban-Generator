import torch
import json
import os
import argparse
from models.resnet import SokobanSEResNetRegressor

def export_fold_model(fold: int):
    # Siempre exportamos en CPU. El binario C++ se encarga de moverlo a GPU si está disponible.
    device = torch.device("cpu")
    print(f"Exporting Fold {fold} on device: {device} (C++ will move to GPU if available)")

    print("Loading hyperparameters...")
    with open("results/best_hparams.json", "r") as f:
        r_params = json.load(f)

    print(f"Loading Regressor Model for Fold {fold}...")
    regressor = SokobanSEResNetRegressor(dropout_p=r_params['params']["dropout_p"])
    
    model_path = f"results/final_regressor_fold{fold}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo {model_path}")
        
    regressor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    regressor.eval()

    print(f"Exporting Regressor Stats for Fold {fold}...")
    stats_path = f"results/regressor_fold{fold}_stats.pt"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"No se encontraron los stats {stats_path}")
        
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    with open("results/surrogate_stats.txt", "w") as sf:
        sf.write(f"{stats['pushes_mean']}\n{stats['pushes_std']}\n")
    print(f"  pushes_mean={stats['pushes_mean']:.4f}, pushes_std={stats['pushes_std']:.4f}")

    print("Tracing model with dummy input (1, 6, 25, 25) on CPU...")
    dummy_input = torch.randn(1, 6, 25, 25)
    
    print("Saving TorchScript model (traced, no freeze)...")
    traced_regressor = torch.jit.trace(regressor, dummy_input)
    
    # Sobrescribimos el JIT que usa el solver C++
    out_jit = "results/surrogate_regressor_jit.pt"
    traced_regressor.save(out_jit)
    
    print(f"\n¡Éxito! El modelo Fold {fold} se exportó a {out_jit}")
    print("El binario de C++ ahora usará este modelo en la próxima ejecución.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Exporta el modelo de un fold a JIT para A*")
    parser.add_argument("--fold", type=int, default=2, help="Número de fold a exportar (1-5). Default: 2")
    args = parser.parse_args()
    
    export_fold_model(args.fold)
