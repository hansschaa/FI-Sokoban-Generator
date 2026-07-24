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

# Statistics for un-normalizing
pushes_mean = 0.0
pushes_std = 1.0
branch_mean = 0.0
branch_std = 1.0

def load_models():
    global classifier_model, regressor_model
    global pushes_mean, pushes_std, branch_mean, branch_std

    # Load Hyperparameters
    with open("surrogate_models/results/best_hparams_classifier.json", "r") as f:
        c_params = json.load(f)
    with open("surrogate_models/results/best_hparams.json", "r") as f:
        r_params = json.load(f)

    # Initialize models
    print(f"Loading Classifier (dropout={c_params['params']['dropout_p']}) to {device}")
    classifier_model = SokobanSEResNetClassifier(dropout_p=c_params['params']["dropout_p"]).to(device)
    classifier_model.load_state_dict(torch.load("surrogate_models/results/final_classifier_fold5.pt", map_location=device))
    classifier_model.eval()

    print(f"Loading Regressor (dropout={r_params['params']['dropout_p']}) to {device}")
    regressor_model = SokobanSEResNetRegressor(dropout_p=r_params['params']["dropout_p"]).to(device)
    regressor_model.load_state_dict(torch.load("surrogate_models/results/final_regressor_fold3.pt", map_location=device))
    regressor_model.eval()

    # Load Regressor stats to un-normalize predictions
    stats = torch.load("surrogate_models/results/regressor_fold3_stats.pt", map_location="cpu", weights_only=False)
    pushes_mean = stats["pushes_mean"]
    pushes_std = stats["pushes_std"]
    branch_mean = stats["branch_mean"]
    branch_std = stats["branch_std"]
    print("Models loaded successfully! Server ready.")

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        if not data or 'boards' not in data:
            return jsonify({"error": "Missing 'boards' array"}), 400
        
        boards_str = data['boards']
        if not boards_str:
            return jsonify([])

        # 1. Encode boards to tensor
        tensors = []
        for b_str in boards_str:
            t_np = encode_board(b_str)
            tensors.append(torch.from_numpy(t_np))
        
        batch_tensor = torch.stack(tensors).to(device)

        # 2. Run Classifier
        with torch.no_grad():
            logits = classifier_model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(-1)
        
        # Determine solvability (threshold 0.5)
        is_solvable = (probs >= 0.5)

        # 3. Run Regressor only on solvable boards (for speed)
        solvable_indices = is_solvable.nonzero(as_tuple=True)[0]
        
        results = [{"is_solvable": False, "pushes": 0.0, "branching": 0.0} for _ in range(len(boards_str))]

        if len(solvable_indices) > 0:
            solvable_tensors = batch_tensor[solvable_indices]
            with torch.no_grad():
                p_norm_pred = regressor_model(solvable_tensors)
                p_norm_pred = p_norm_pred.squeeze(-1)
            
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

if __name__ == '__main__':
    load_models()
    # Run on port 5000, only local connections
    app.run(host='127.0.0.1', port=5000, threaded=False)
