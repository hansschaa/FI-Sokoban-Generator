import os
import sys
import json
import torch
import numpy as np
import re
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'surrogate_models'))
from data.board_utils import encode_board
from models.resnet import SokobanSEResNetRegressor

def parse_sok_file(fpath):
    records = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2: continue

        header = lines[0]
        board_lines = lines[1:]

        m_pushes = re.search(r"pushes:(\d+)", header)
        if m_pushes: pushes = int(m_pushes.group(1))
        else: continue
        
        import hashlib
        
        board_str = "\n".join(board_lines)
        MOBILE_CHARS = str.maketrans("$.*@+", "     ")
        shell_str = board_str.translate(MOBILE_CHARS)
        shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()

        records.append({
            "board_str": board_str,
            "pushes": pushes,
            "shell_hash": shell_hash
        })
    return records

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load fold map
    fold_map_path = "surrogate_models/results/fold_map_v2.json"
    with open(fold_map_path, "r") as f:
        fold_map = json.load(f)
    
    # We want fold 3 dense boards
    fold3_hashes = set()
    for h, fold in fold_map.items():
        if fold == 3:
            fold3_hashes.add(h)
    
    print(f"Found {len(fold3_hashes)} dense shell_hashes in Fold 3.")

    # 2. Parse DenseSolvables to find the actual boards
    dense_dir = "training_data/DenseSolvables"
    print(f"Parsing {dense_dir}...")
    dense_records = []
    for fpath in glob.glob(os.path.join(dense_dir, "*.sok")):
        recs = parse_sok_file(fpath)
        for r in recs:
            if r["shell_hash"] in fold3_hashes:
                dense_records.append(r)
    
    print(f"Found {len(dense_records)} dense boards for Fold 3 test set.")

    if len(dense_records) == 0:
        print("No dense boards found. Exiting.")
        return

    # 3. Encode boards to tensors
    print("Encoding boards...")
    test_data = []
    for r in dense_records:
        t = encode_board(r["board_str"])
        test_data.append({
            "tensor": torch.tensor(t, dtype=torch.float32),
            "pushes_raw": float(r["pushes"]),
            "shell_hash": r["shell_hash"]
        })

    # 4. Load the 100% Production Model
    # Note: The user said we should evaluate MAE for Fold 3 model.
    model_path = "surrogate_models/results/path_consistency/production_path_consistency.pt"
    print(f"Loading model: {model_path}")
    model = SokobanSEResNetRegressor(dropout_p=0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Normalize stats for PC model
    p_mean = 3.4614
    p_std = 0.8732

    # 5. Evaluate MAE
    all_preds = []
    all_y = []
    
    print("Running inference...")
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            X = torch.stack([b["tensor"] for b in batch]).to(device)
            y_raw = np.array([b["pushes_raw"] for b in batch])
            
            preds_norm = model(X).squeeze(-1)
            preds_raw = torch.expm1(preds_norm * p_std + p_mean).cpu().numpy()
            
            all_preds.extend(preds_raw)
            all_y.extend(y_raw)
            
    all_preds = np.array(all_preds)
    all_y = np.array(all_y)
    
    mae = np.mean(np.abs(all_preds - all_y))
    print(f"\n[Dataset Denso (Fold 3)] N={len(all_y):,} | MAE: {mae:.2f} pushes")

if __name__ == "__main__":
    main()
