import os
import torch
from collections import defaultdict
from tqdm import tqdm

def main():
    dense_dir = "DensePathConsistency"
    orig_dir = "OriginalPathConsistency"
    out_dir = "surrogate_models/results/path_consistency_v2"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Count original pairs
    print("Calculando conteo de pares originales...")
    orig_files = defaultdict(list)
    for f in os.listdir(orig_dir):
        if f.startswith("path_fold") and f.endswith(".pt"):
            k = int(f.split("fold")[1].split("_")[0])
            orig_files[k].append(os.path.join(orig_dir, f))
            
    orig_counts = {}
    for k in range(1, 6):
        total_orig_k = 0
        for f in orig_files[k]:
            data = torch.load(f, map_location='cpu', weights_only=False)
            total_orig_k += len(data)
        orig_counts[k] = total_orig_k
            
    print("--- CONTEO ORIGINAL ---")
    for k in range(1, 6):
        print(f"Fold {k}: {orig_counts[k]:,} pares")
        
    print("\nCalculando conteo de pares densos...")
    dense_files = defaultdict(list)
    for f in os.listdir(dense_dir):
        if f.startswith("path_fold") and f.endswith(".pt"):
            # Extract fold k
            # format: path_fold{k}_train_part{p}.pt
            k = int(f.split("fold")[1].split("_")[0])
            dense_files[k].append(os.path.join(dense_dir, f))
            
    dense_counts = {}
    for k in range(1, 6):
        total_dense_k = 0
        for f in dense_files[k]:
            data = torch.load(f, map_location='cpu', weights_only=False)
            total_dense_k += len(data)
        dense_counts[k] = total_dense_k
        
    print("\n--- BALANCE ESTIMADO POST-FUSIÓN ---")
    for k in range(1, 6):
        total = orig_counts.get(k, 0) + dense_counts.get(k, 0)
        orig_pct = (orig_counts.get(k, 0) / total * 100) if total > 0 else 0
        dense_pct = (dense_counts.get(k, 0) / total * 100) if total > 0 else 0
        print(f"Fold {k}: Total = {total:,} | Original = {orig_counts.get(k, 0):,} ({orig_pct:.1f}%) | Denso = {dense_counts.get(k, 0):,} ({dense_pct:.1f}%)")

    print("\nPara ejecutar la fusión real, descomente el código de guardado en el script.")
    
    # 2. Merge (Commented out to allow user verification first)
    
    for k in range(1, 6):
        print(f"\nFusionando Fold {k}...")
        merged_data = []
        
        # Load original parts
        for f in orig_files[k]:
            merged_data.extend(torch.load(f, map_location='cpu', weights_only=False))
            
        # Load dense parts
        for f in tqdm(dense_files[k], desc=f"Particiones densas Fold {k}"):
            merged_data.extend(torch.load(f, map_location='cpu', weights_only=False))
            
        out_path = os.path.join(out_dir, f"path_fold{k}_train.pt")
        print(f"Guardando Fold {k} unificado en {out_path} ({len(merged_data)} pares)...")
        torch.save(merged_data, out_path)
        
    print("\n¡Fusión completada!")
    

if __name__ == "__main__":
    main()
