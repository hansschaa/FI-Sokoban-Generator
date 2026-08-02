import os
import torch
import numpy as np
import pickle
import sys

# Append surrogate_models to sys.path so we can import get_hungarian_lb and the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate_models"))
from surrogate_server import get_hungarian_lb
from models.resnet import SokobanSEResNetRegressor
from torch.utils.data import DataLoader, Dataset

TRAIN_PT = "surrogate_models/results/siamese_ranknet_train.pt"
TEST_PT = "surrogate_models/results/siamese_ranknet_test_heldout.pt"
STATS_PATH = "surrogate_models/results/production_regressor_stats.pt"
PROD_MODEL_PATH = "surrogate_models/results/production_regressor.pt"
CALIB_PKL = "surrogate_models/results/regressor_calibration.pkl"

class SiameseDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        item = self.pairs[idx]
        return item["tensor_A"], item["raw_A"], item["tensor_B"], item["raw_B"]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    train_pairs = torch.load(TRAIN_PT, weights_only=False)
    test_pairs = torch.load(TEST_PT, weights_only=False)
    
    loader = DataLoader(SiameseDataset(train_pairs + test_pairs), batch_size=256, shuffle=False)
    
    stats = torch.load(STATS_PATH, weights_only=False)
    pushes_mean = stats["pushes_mean"]
    pushes_std = stats["pushes_std"]
    
    with open("surrogate_models/results/best_hparams.json", "r") as f:
        import json
        r_params = json.load(f)
    
    model = SokobanSEResNetRegressor(dropout_p=r_params['params']['dropout_p']).to(device)
    model.load_state_dict(torch.load(PROD_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    with open(CALIB_PKL, "rb") as f:
        regressor_calibration = pickle.load(f)
        
    k_reqs = []
    
    print("Evaluating ~60k boards to find the 95th percentile clip multiplier...")
    with torch.no_grad():
        for tA, rA, tB, rB in loader:
            pA_norm = model(tA.to(device))
            pB_norm = model(tB.to(device))
            
            pA_raw = np.expm1((pA_norm.cpu().numpy() * pushes_std) + pushes_mean).flatten()
            pB_raw = np.expm1((pB_norm.cpu().numpy() * pushes_std) + pushes_mean).flatten()
            
            pA_calib = regressor_calibration.predict(pA_raw)
            pB_calib = regressor_calibration.predict(pB_raw)
            
            # For each tensor A
            for i in range(len(tA)):
                lb = get_hungarian_lb(tA[i])
                raw = pA_raw[i]
                calib = pA_calib[i]
                if calib > raw and lb > 0:
                    k = (calib - lb) / lb
                    k_reqs.append(k)
                    
            # For each tensor B
            for i in range(len(tB)):
                lb = get_hungarian_lb(tB[i])
                raw = pB_raw[i]
                calib = pB_calib[i]
                if calib > raw and lb > 0:
                    k = (calib - lb) / lb
                    k_reqs.append(k)
                    
    k_reqs = np.array(k_reqs)
    print(f"Total boards with upward corrections (and lb > 0): {len(k_reqs)}")
    
    if len(k_reqs) > 0:
        k_95 = np.percentile(k_reqs, 95)
        k_99 = np.percentile(k_reqs, 99)
        k_max = np.max(k_reqs)
        
        print(f"95th Percentile K: {k_95:.4f}")
        print(f"99th Percentile K: {k_99:.4f}")
        print(f"Max K observed:    {k_max:.4f}")
        
        # Save the chosen K to a file so we can read it easily
        with open("scratch/chosen_k.txt", "w") as f:
            f.write(str(k_95))
    else:
        print("No upward corrections found?!")

if __name__ == "__main__":
    main()
