import os
import json
import torch
import traceback
from flask import Flask, request, jsonify
from models.resnet import SokobanSEResNetClassifier, SokobanSEResNetRegressor
from data.prepare_classifier import encode_board

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global models
classifier_model = None
regressor_model = None

# Statistics for un-normalizing and classification threshold
pushes_mean = 0.0
pushes_std = 1.0
CLASSIFIER_THRESHOLD = 0.65

def load_models():
    global classifier_model, regressor_model
    global pushes_mean, pushes_std, CLASSIFIER_THRESHOLD

    import hashlib
    
    def compute_sha256(filepath):
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # Load Hyperparameters & Calibrated Threshold
    c_hparams_path = "surrogate_models/results/best_hparams_contrastive_classifier.json"
    if os.path.exists(c_hparams_path):
        with open(c_hparams_path, "r", encoding="utf-8") as f:
            c_config = json.load(f)
        c_params_dict = c_config.get("best_params", c_config.get("params", {}))
        c_dropout = float(c_params_dict.get("dropout_p", 0.4))
        CLASSIFIER_THRESHOLD = float(c_config.get("optimal_threshold", 0.70))
        print(f"✅ Loaded contrastive config from {c_hparams_path} (dropout={c_dropout:.4f}, threshold={CLASSIFIER_THRESHOLD:.2f})")
    else:
        with open("surrogate_models/results/best_hparams_classifier.json", "r", encoding="utf-8") as f:
            c_params = json.load(f)
        c_dropout = float(c_params["params"]["dropout_p"])
        CLASSIFIER_THRESHOLD = 0.65
        print(f"⚠️ {c_hparams_path} not found. Fallback to default threshold {CLASSIFIER_THRESHOLD} and dropout {c_dropout}")

    with open("surrogate_models/results/best_hparams.json", "r", encoding="utf-8") as f:
        r_params = json.load(f)

    # Initialize models
    c_path = "surrogate_models/results/production_contrastive_classifier.pt"
    if not os.path.exists(c_path):
        fallback_path = "surrogate_models/results/final_contrastive_classifier_fold1.pt"
        if os.path.exists(fallback_path):
            print(f"⚠️ Production model not found at {c_path}. Falling back to {fallback_path}")
            c_path = fallback_path
    print(f"Loading Classifier (dropout={c_dropout}) to {device}")
    print(f"Model Path: {c_path}")
    print(f"Model SHA256: {compute_sha256(c_path)}")
    classifier_model = SokobanSEResNetClassifier(dropout_p=c_dropout, in_channels=12).to(device)
    classifier_model.load_state_dict(torch.load(c_path, map_location=device))
    classifier_model.eval()

    r_path = "surrogate_models/results/production_regressor.pt"
    print(f"Loading Regressor (dropout={r_params['params']['dropout_p']}) to {device}")
    print(f"Model Path: {r_path}")
    print(f"Model SHA256: {compute_sha256(r_path)}")
    regressor_model = SokobanSEResNetRegressor(dropout_p=r_params['params']["dropout_p"]).to(device)
    regressor_model.load_state_dict(torch.load(r_path, map_location=device))
    regressor_model.eval()

    # Load Regressor stats to un-normalize predictions
    stats = torch.load("surrogate_models/results/production_regressor_stats.pt", map_location="cpu", weights_only=False)
    pushes_mean = stats["pushes_mean"]
    pushes_std = stats["pushes_std"]
    
    if device.type == "cuda":
        print(f"GPU Device Name: {torch.cuda.get_device_name(device)}")
    print("Surrogate Parallelism: Batched GPU Inference (batch size = dynamic, up to population size)")
    print("Models loaded successfully! Server ready.")

import numpy as np

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        if not data or 'boards' not in data:
            return jsonify({"error": "Missing 'boards' array"}), 400
        
        boards_data = data['boards']
        if not boards_data:
            return jsonify([])

        in_channels = getattr(classifier_model, 'stem', None)
        if in_channels is not None:
            in_channels = classifier_model.stem[0].in_channels
        else:
            in_channels = classifier_model.conv1.in_channels
        # 1. Encode boards to tensor
        tensors = []
        for item in boards_data:
            if not isinstance(item, dict):
                return jsonify({"error": "Items must be dicts with 'board' and 'parent_board'"}), 400
                
            b_str = item["board"]
            p_str = item["parent_board"]
            t_b = encode_board(b_str)
            if in_channels == 12:
                t_p = encode_board(p_str)
                t_np = np.concatenate([t_p, t_b], axis=0)
            else:
                t_np = t_b
            tensors.append(torch.from_numpy(t_np))
        
        batch_tensor = torch.stack(tensors).to(device)

        # 2. Run Classifier
        with torch.no_grad():
            logits = classifier_model(batch_tensor)
            probs = torch.sigmoid(logits)
        
        # Determine solvability using dynamically loaded calibrated threshold
        is_solvable = (probs >= CLASSIFIER_THRESHOLD)
        
        # Log to file for verification
        with open("scratch/probs.log", "a") as f:
            for p in probs.tolist():
                f.write(f"{p:.6f}\n")

        # 3. Run Regressor only on solvable boards (for speed)
        solvable_indices = is_solvable.nonzero(as_tuple=True)[0]
        
        results = [{"is_solvable": False, "pushes": 0.0, "branching": 0.0} for _ in range(len(boards_data))]

        if len(solvable_indices) > 0:
            solvable_tensors = batch_tensor[solvable_indices]
            # The regressor was not retrained and still expects 6 channels (only the current board)
            # Since batch_tensor is [parent, child], the current board is in the last 6 channels
            if in_channels == 12:
                solvable_tensors_reg = solvable_tensors[:, 6:, :, :]
            else:
                solvable_tensors_reg = solvable_tensors
                
            with torch.no_grad():
                p_norm_pred = regressor_model(solvable_tensors_reg)
            
            # Un-normalize
            p_pred_log = (p_norm_pred * pushes_std) + pushes_mean
            p_pred = torch.clamp(torch.expm1(p_pred_log), min=0.0)

            # Map back to results
            for i, idx in enumerate(solvable_indices.tolist()):
                results[idx]["is_solvable"] = True
                results[idx]["pushes"] = p_pred[i].item()
                results[idx]["branching"] = 1.0

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate_regressor_only', methods=['POST'])
def evaluate_regressor_only():
    try:
        data = request.get_json()
        if not data or 'boards' not in data:
            return jsonify({"error": "Missing 'boards' array"}), 400
        
        boards_data = data['boards']
        if not boards_data:
            return jsonify([])

        in_channels = getattr(regressor_model, 'stem', None)
        if in_channels is not None:
            in_channels = regressor_model.stem[0].in_channels
        else:
            in_channels = regressor_model.conv1.in_channels
            
        # USER REQUESTED CHECK: ensure the tensor for the regressor is exactly 6 channels
        if in_channels != 6:
            return jsonify({"error": f"CRITICAL: Regressor expected 6 channels but model has {in_channels}"}), 500

        tensors = []
        for item in boards_data:
            if not isinstance(item, dict):
                return jsonify({"error": "Items must be dicts with 'board'"}), 400
                
            b_str = item["board"]
            # Regressor only takes the current board (6 channels)
            t_b = encode_board(b_str)
            tensors.append(torch.from_numpy(t_b))
        
        batch_tensor = torch.stack(tensors).to(device)

        # Run Regressor directly on all provided boards
        results = [{"is_solvable": True, "pushes": 0.0, "branching": 0.0} for _ in range(len(boards_data))]
        
        with torch.no_grad():
            p_norm_pred = regressor_model(batch_tensor)
            
            # Un-normalize
            p_pred_log = (p_norm_pred * pushes_std) + pushes_mean
            p_pred = torch.clamp(torch.expm1(p_pred_log), min=0.0)

            # Map back to results
            for i in range(len(boards_data)):
                results[i]["pushes"] = p_pred[i].item()
                results[i]["branching"] = 1.0

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    # Run on port 5000, only local connections
    app.run(host='127.0.0.1', port=5000, threaded=False)
