import os
import torch
import numpy as np
import pickle
from sklearn.isotonic import IsotonicRegression

TRAIN_PT = "surrogate_models/results/siamese_ranknet_train.pt"
TEST_PT = "surrogate_models/results/siamese_ranknet_test_heldout.pt"
STATS_PATH = "surrogate_models/results/production_regressor_stats.pt"
PROD_MODEL_PATH = "surrogate_models/results/production_regressor.pt"
OUT_PKL = "surrogate_models/results/regressor_calibration.pkl"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate_models"))
from models.resnet import SokobanSEResNetRegressor
from torch.utils.data import DataLoader, Dataset

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
    
    all_pred = []
    all_real = []
    
    print("Extracting raw predictions vs real pushes on ~60k boards...")
    with torch.no_grad():
        for tA, rA, tB, rB in loader:
            pA_norm = model(tA.to(device))
            pB_norm = model(tB.to(device))
            
            pA_raw = np.expm1((pA_norm.cpu().numpy() * pushes_std) + pushes_mean)
            pB_raw = np.expm1((pB_norm.cpu().numpy() * pushes_std) + pushes_mean)
            
            all_pred.extend(pA_raw.tolist())
            all_pred.extend(pB_raw.tolist())
            all_real.extend(rA.numpy().tolist())
            all_real.extend(rB.numpy().tolist())
            
    all_pred = np.array(all_pred)
    all_real = np.array(all_real)
    
    print("Fitting Isotonic Regression...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(all_pred, all_real)
    
    print("\n=== CALIBRATION DIAGNOSTICS ===")
    test_points = [10, 20, 30, 40, 50, 75, 100, 150]
    for p in test_points:
        calib_p = iso.predict([p])[0]
        diff = calib_p - p
        print(f"Raw Pred: {p:3d} -> Calibrated: {calib_p:6.2f} (Delta: {diff:+.2f})")
        
    print(f"\nCalibration Min Pred: {iso.X_min_:.2f} | Max Pred: {iso.X_max_:.2f}")
    
    with open(OUT_PKL, "wb") as f:
        pickle.dump(iso, f)
    print(f"Saved calibration to {OUT_PKL}")

if __name__ == "__main__":
    main()
