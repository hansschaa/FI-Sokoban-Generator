import torch
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../surrogate_models")
from models.resnet import SokobanSEResNetRegressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
    
    # Load state dict
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Sometimes saved as dict with 'model_state_dict', sometimes just the state_dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
        
    model.eval()

    # Create dummy input: batch_size=1, channels=6, width=12, height=12
    # The regressor takes 6 channels per state (even for path consistency, it calculates pred1 and pred2 independently)
    # Wait, the dataset uses 10x10 boards, but it's padded. What size does C++ send?
    # Usually max size is 12x12 or 10x10. Let's trace it dynamically or with a large enough max size.
    # Actually, ResNet uses AdaptiveAvgPool2d so spatial dimension doesn't matter for tracing, but we trace with a standard size.
    dummy_input = torch.zeros(1, 6, 20, 20).to(device)
    
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(args.output_path)
    
    import hashlib
    with open(args.output_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    
    print(f"✅ Exported to {args.output_path}")
    print(f"🔑 SHA256 Checksum: {sha256}")

if __name__ == "__main__":
    main()
