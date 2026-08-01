import os
import torch
import sys

device = torch.device("cpu")
test_data = torch.load("surrogate_models/results/regressor_fold1_test.pt", weights_only=False)

from collections import defaultdict

shells = defaultdict(list)
for item in test_data:
    num_boxes = item.get('num_boxes', 0)
    if num_boxes == 0:
        num_boxes = int(item['tensor'][2].sum().item())
    shells[item['shell_hash']].append((num_boxes, item['pushes_raw']))

print("=== Ejemplos grupo 'same_count' (100% misma cantidad en todo el shell) ===")
same_found = 0
for shell_hash, variants in shells.items():
    if len(variants) < 3: continue
    box_counts = set(v[0] for v in variants)
    if len(box_counts) == 1:
        print(shell_hash, "-> num_boxes:", [v[0] for v in variants])
        same_found += 1
        if same_found >= 3: break
if same_found == 0:
    print("(Ningún shell con >=3 variantes tiene una cantidad de cajas única)")

print("\n=== Ejemplos grupo 'diff_count' (varias cantidades distintas en el shell) ===")
diff_found = 0
for shell_hash, variants in shells.items():
    if len(variants) < 3: continue
    box_counts = set(v[0] for v in variants)
    if len(box_counts) > 1:
        print(shell_hash, "-> num_boxes:", [v[0] for v in variants][:15], "..." if len(variants)>15 else "")
        diff_found += 1
        if diff_found >= 3: break
