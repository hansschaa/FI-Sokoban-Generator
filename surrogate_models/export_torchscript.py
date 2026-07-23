import torch
import json
import os
from models.resnet import SokobanResNetRegressor, SokobanResNetClassifier

def export_model():
    print("Loading hyperparameters...")
    with open("results/best_hparams.json", "r") as f:
        r_params = json.load(f)
    
    with open("results/best_hparams_classifier.json", "r") as f:
        c_params = json.load(f)

    print("Loading Regressor Model...")
    regressor = SokobanResNetRegressor(dropout_p=r_params['params']["dropout_p"])
    regressor.load_state_dict(torch.load("results/final_regressor_fold3.pt", map_location="cpu"))
    regressor.eval()

    print("Loading Classifier Model...")
    classifier = SokobanResNetClassifier(dropout_p=c_params['params']["dropout_p"])
    classifier.load_state_dict(torch.load("results/final_classifier_fold5.pt", map_location="cpu"))
    classifier.eval()

    print("Tracing models with dummy input (1, 5, 25, 25)...")
    dummy_input = torch.randn(1, 5, 25, 25)
    
    traced_regressor = torch.jit.trace(regressor, dummy_input)
    traced_classifier = torch.jit.trace(classifier, dummy_input)

    print("Saving TorchScript models...")
    traced_regressor.save("results/surrogate_regressor_jit.pt")
    traced_classifier.save("results/surrogate_classifier_jit.pt")
    
    print("Successfully exported models to TorchScript!")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    export_model()
