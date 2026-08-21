import os
import torch
from collections import defaultdict
from tqdm import tqdm
import gc

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
        
        t1s, t2s, p1s, p2s, hashes = [], [], [], [], []
        
        def process_part(fpath):
            data = torch.load(fpath, map_location='cpu', weights_only=False)
            for item in data:
                # Some old data might have numpy arrays, convert to tensor
                t1 = item['tensor1']
                t2 = item['tensor2']
                if not isinstance(t1, torch.Tensor):
                    t1 = torch.tensor(t1, dtype=torch.float32)
                if not isinstance(t2, torch.Tensor):
                    t2 = torch.tensor(t2, dtype=torch.float32)
                
                t1s.append(t1)
                t2s.append(t2)
                p1s.append(item['pushes1'])
                p2s.append(item['pushes2'])
                hashes.append(item.get('shell_hash', ''))
                
        # Load original parts
        for f in orig_files[k]:
            process_part(f)
            
        # Load dense parts
        for f in tqdm(dense_files[k], desc=f"Particiones densas Fold {k}"):
            process_part(f)
            
        print(f"Apilando tensores del Fold {k}...")
        out_dict = {
            'tensor1': torch.stack(t1s).byte() if t1s[0].dtype == torch.uint8 else torch.stack(t1s),
            'tensor2': torch.stack(t2s).byte() if t2s[0].dtype == torch.uint8 else torch.stack(t2s),
            'pushes1': torch.tensor(p1s, dtype=torch.int16),
            'pushes2': torch.tensor(p2s, dtype=torch.int16),
            'shell_hash': hashes
        }
        
        out_path = os.path.join(out_dir, f"path_fold{k}_train.pt")
        print(f"Guardando Fold {k} unificado estructurado en {out_path} ({len(p1s)} pares)...")
        torch.save(out_dict, out_path)
        
        del t1s, t2s, p1s, p2s, hashes, out_dict
        gc.collect()
        
    print("\n¡Fusión completada!")
    

if __name__ == "__main__":
    main()
