import torch
path = "surrogate_models/results/path_consistency/path_fold1_train.pt"
print(f"Loading {path}...")
pairs = torch.load(path, weights_only=False, map_location='cpu')
print(f"Total pairs: {len(pairs)}")
print("First 5 pairs pushes1 vs pushes2:")
for i in range(min(5, len(pairs))):
    print(f"  Pair {i}: pushes1={pairs[i]['pushes1']}, pushes2={pairs[i]['pushes2']}")
