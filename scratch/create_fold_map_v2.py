import json
import os
import glob
import re
import hashlib
import random

def create_fold_map_v2():
    # Load original
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, "surrogate_models", "results", "fold_map.json")
    
    with open(orig_path, "r") as f:
        fold_map = json.load(f)
        
    print(f"Loaded {len(fold_map)} original shell_hashes.")
    
    # Read dense
    dense_dir = os.path.join(base_dir, "training_data", "DenseSolvables")
    dense_files = glob.glob(os.path.join(dense_dir, "**/*.sok"), recursive=True)
    
    dense_hashes = set()
    
    for fpath in dense_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3: continue
            board_str = "\n".join(lines[1:])
            MOBILE_CHARS = str.maketrans("$.*@+", "     ")
            shell_str = board_str.translate(MOBILE_CHARS)
            shell_hash = hashlib.sha256(shell_str.encode()).hexdigest()
            dense_hashes.add(shell_hash)
            
    print(f"Found {len(dense_hashes)} unique dense shell_hashes.")
    
    # Assign dense to folds 1-5 evenly
    dense_list = sorted(list(dense_hashes))
    random.seed(42) # Deterministic assignment
    random.shuffle(dense_list)
    
    for i, h in enumerate(dense_list):
        if h not in fold_map: # Just in case
            fold_map[h] = (i % 5) + 1
            
    out_path = os.path.join(base_dir, "surrogate_models", "results", "fold_map_v2.json")
    with open(out_path, "w") as f:
        json.dump(fold_map, f)
        
    print(f"Saved {len(fold_map)} total shell_hashes to {out_path}.")
    
    # Report
    counts = {k: {"Original": 0, "Denso": 0} for k in range(1, 6)}
    
    # Recalculate which is which
    orig_hashes = set()
    with open(orig_path, "r") as f:
        orig_map = json.load(f)
        orig_hashes = set(orig_map.keys())
        
    for h, fold in fold_map.items():
        if h in orig_hashes:
            counts[fold]["Original"] += 1
        else:
            counts[fold]["Denso"] += 1
            
    print("\n[!] Distribución de shell_hashes por Fold:")
    print("Fold\tOriginal\tDenso\t\tTotal")
    print("-" * 50)
    for k in range(1, 6):
        o = counts[k]["Original"]
        d = counts[k]["Denso"]
        print(f"{k}\t{o}\t\t{d}\t\t{o+d}")
        
if __name__ == "__main__":
    create_fold_map_v2()
