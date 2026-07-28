import torch
import json
import os
from models.resnet import SokobanSEResNetRegressor, SokobanSEResNetClassifier

def export_model():
    # Siempre exportamos en CPU. El binario C++ se encarga de moverlo a GPU si está disponible.
    # NO usar freeze: un modelo congelado embebe los pesos como constantes
    # inmutables que no pueden moverse entre CPU/CUDA.
    device = torch.device("cpu")
    print(f"Exporting on device: {device} (C++ will move to GPU if available)")

    print("Loading hyperparameters...")
    with open("results/best_hparams.json", "r") as f:
        r_params = json.load(f)
    
    with open("results/best_hparams_classifier.json", "r") as f:
        c_params = json.load(f)

    print("Loading Regressor Model (Production)...")
    regressor = SokobanSEResNetRegressor(dropout_p=r_params['params']["dropout_p"])
    regressor.load_state_dict(torch.load("results/production_regressor.pt", map_location="cpu", weights_only=True))
    regressor.eval()

    print("Exporting Regressor Stats...")
    stats = torch.load("results/production_regressor_stats.pt", map_location="cpu", weights_only=True)
    with open("results/surrogate_stats.txt", "w") as sf:
        sf.write(f"{stats['pushes_mean']}\n{stats['pushes_std']}\n")

    print("Loading Classifier Model (Production)...")
    classifier = SokobanSEResNetClassifier(dropout_p=c_params['params']["dropout_p"])
    classifier.load_state_dict(torch.load("results/production_classifier.pt", map_location="cpu", weights_only=True))
    classifier.eval()

    print("Tracing models with dummy input (1, 6, 25, 25) on CPU...")
    dummy_input = torch.randn(1, 6, 25, 25)
    
    print("Saving TorchScript models (traced, no freeze)...")
    traced_regressor = torch.jit.trace(regressor, dummy_input)
    traced_regressor.save("results/surrogate_regressor_jit.pt")
    
    traced_classifier = torch.jit.trace(classifier, dummy_input)
    traced_classifier.save("results/surrogate_classifier_jit.pt")
    
    print("Successfully exported models to TorchScript (CPU, moveable to GPU at runtime)!")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    export_model()
