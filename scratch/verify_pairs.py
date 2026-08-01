import torch
import sys

def verify():
    path = "surrogate_models/results/path_consistency/path_fold1_train.pt"
    try:
        ds = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Total pairs in {path}: {len(ds)}")
        
        pairs = set([(p['pushes1'], p['pushes2']) for p in ds])
        print(f"Unique (pushes1, pushes2) pairs: {len(pairs)}")
        print("Sample of 10 unique pairs:")
        for i, pair in enumerate(list(pairs)[:10]):
            print(f"  {pair}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify()
