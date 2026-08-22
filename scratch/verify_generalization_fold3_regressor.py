import sys, os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'surrogate_models'))
from models.resnet import SokobanSEResNetRegressor
from train_final_path_consistency import RegressorDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="surrogate_models/results/path_consistency/production_path_consistency.pt")
    parser.add_argument("--test-data", type=str, default="surrogate_models/results/regressor_fold3_test.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading test data from: {args.test_data}")
    try:
        test_data_raw = torch.load(args.test_data, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return
        
    model = SokobanSEResNetRegressor(dropout_p=0.1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()

    # The dataset RegressorDataset expects a list of dicts. We need to extract 's' manually if we can.
    # If the dict doesn't have 's' directly, we can check if it has 'origin' or deduce it.
    all_s = []
    has_s = True
    for item in test_data_raw:
        if 's' in item:
            all_s.append(item['s'])
        elif 'origin' in item:
            all_s.append(1 if item['origin'] == 'dense' else 0)
        else:
            has_s = False
            break
            
    if not has_s:
        print("Warning: Field 's' or 'origin' not found in test dataset dictionaries. Inferring from wall count...")
        all_s = []
        for item in test_data_raw:
            # item['tensor'] shape is [6, 10, 10], channel 0 is walls
            walls = item['tensor'][0].sum().item()
            # Dense boards have very few walls (usually < 22), Original have more (around 25-35)
            if walls < 22:
                all_s.append(1) # Dense
            else:
                all_s.append(0) # Original
        has_s = True
        
    all_s = np.array(all_s)

    dataset = RegressorDataset(test_data_raw)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_y = []

    print("Evaluating...")
    with torch.no_grad():
        # Loader yields (tensor, pushes_norm, pushes_raw, weight)
        for X, _, y_raw, _ in loader:
            X = X.to(device)
            preds_norm = model(X).squeeze(-1)
            # Denormalize (using the fixed stats for Fold 1/3)
            preds_raw = torch.expm1(preds_norm * 0.8732 + 3.4614)
            all_preds.append(preds_raw.cpu())
            all_y.append(y_raw)

    all_preds = torch.cat(all_preds).numpy()
    all_y = torch.cat(all_y).numpy()

    def print_mae(name, mask):
        y_true_m = all_y[mask]
        y_pred_m = all_preds[mask]
        if len(y_true_m) == 0:
            print(f"[{name}] No samples.")
            return
        mae = np.mean(np.abs(y_true_m - y_pred_m))
        print(f"[{name}] N={len(y_true_m):,} | MAE: {mae:.2f} pushes")

    if has_s:
        mask_s0 = (all_s == 0)
        mask_s1 = (all_s == 1)
        print_mae("Dataset Original (s==0)", mask_s0)
        print_mae("Dataset Denso (s==1)", mask_s1)
    
    print_mae("Combinado (All)", np.ones_like(all_y, dtype=bool))

if __name__ == "__main__":
    main()
