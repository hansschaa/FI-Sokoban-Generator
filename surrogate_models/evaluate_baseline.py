import torch
import sys
import os
from models.resnet import SokobanSEResNetRegressor
from evaluate_inter_branch import get_valid_children

def evaluate_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SokobanSEResNetRegressor(dropout_p=0.0) # Evaluacion pura
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    candidates = [
        os.path.join(SCRIPT_DIR, "results", "production_regressor.pt"),
        os.path.join(SCRIPT_DIR, "results", "final_regressor_fold1.pt"),
        os.path.join(SCRIPT_DIR, "results", "path_consistency", "final_regressor_fold1.pt")
    ]
    
    ckpt_path = None
    for cand in candidates:
        if os.path.exists(cand):
            ckpt_path = cand
            break
            
    if not ckpt_path:
        print(f"Error: No se encontro el modelo en ninguna de las siguientes rutas:")
        for cand in candidates:
            print(f"  - {cand}")
        return
        
    print(f"Cargando {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    
    sys.path.append(SCRIPT_DIR)
    from optuna_path_consistency import evaluate_model_inter_branch
    acc = evaluate_model_inter_branch(model, device, n_pairs=5000)
    
    print("==================================================")
    print(f"  BASELINE (Old Production Regressor)")
    print(f"  Inter-branch Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("==================================================")

if __name__ == "__main__":
    evaluate_baseline()
