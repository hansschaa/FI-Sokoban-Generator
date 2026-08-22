import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'surrogate_models'))
from models.resnet import SokobanSEResNetRegressor

def get_bucket(pushes):
    if pushes <= 10: return "1_to_10"
    if pushes > 100: return "101_plus"
    lower = ((pushes - 1) // 10) * 10 + 1
    upper = lower + 9
    return f"{lower}_to_{upper}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(BASE_DIR, "..", "surrogate_models", "results", "path_consistency", "production_path_consistency.pt")
    
    print(f"Cargando modelo: {model_path}")
    model = SokobanSEResNetRegressor(dropout_p=0.4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Stats used for normalization during training
    # Based on train_kfold_path_consistency_v2.py
    p_mean = 16.27
    p_std = 12.89

    def evaluate_partition(dir_path, name):
        if not os.path.exists(dir_path):
            print(f"\n[!] No se encontro el directorio: {dir_path}")
            return
            
        print(f"\n=== Evaluando {name} (Fold 3) ===")
        all_mae = []
        buckets = {b: [] for b in ["1_to_10", "11_to_20", "21_to_30", "31_to_40", "41_to_50", "51_to_60", "61_to_70", "71_to_80", "81_to_90", "91_to_100", "101_plus"]}
        
        # En k-fold, el fold de test es 3, o sea path_fold3_train_partX.pt
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith("path_fold3") and f.endswith(".pt")]
        
        if not files:
            print(f"No se encontraron particiones del Fold 3 en {dir_path}")
            return
            
        total_pairs = 0
        with torch.no_grad():
            for fpath in files:
                data = torch.load(fpath, map_location='cpu', weights_only=False)
                for item in data:
                    pushes1 = item['pushes1']
                    pushes2 = item['pushes2']
                    
                    # Filtro max_route_distance=1 (diferencia de empujes == 4, ya que K=4)
                    if abs(pushes1 - pushes2) != 4:
                        continue
                        
                    t1 = item['tensor1']
                    t2 = item['tensor2']
                    if not isinstance(t1, torch.Tensor):
                        t1 = torch.tensor(t1, dtype=torch.float32)
                    if not isinstance(t2, torch.Tensor):
                        t2 = torch.tensor(t2, dtype=torch.float32)
                        
                    # Stack
                    x = torch.stack([t1, t2]).to(device)
                    y_true = abs(pushes1 - pushes2)
                    
                    p_pred, _ = model(x)
                    p_desnorm = (p_pred.cpu().item() * p_std) + p_mean
                    
                    error = abs(p_desnorm - y_true)
                    all_mae.append(error)
                    buckets[get_bucket(pushes2)].append(error)
                    total_pairs += 1
        
        if not all_mae:
            print("No se encontraron pares validos con max_route_distance=1.")
            return
            
        print(f"Total Pares (dist=1): {total_pairs}")
        print(f"MAE Global: {np.mean(all_mae):.4f}")
        print("\nDesglose por dificultad (basado en pushes2):")
        for b, errs in buckets.items():
            if errs:
                print(f"  {b:10s}: {np.mean(errs):.4f} (N={len(errs)})")
                
    evaluate_partition(os.path.join(BASE_DIR, "..", "DensePathConsistency"), "Origen DENSO (s=1)")
    evaluate_partition(os.path.join(BASE_DIR, "..", "OriginalPathConsistency"), "Origen ORIGINAL (s=0)")

if __name__ == "__main__":
    main()
