import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
sys.path.append(os.path.abspath('surrogate_models'))
from models.resnet import SokobanSEResNetRegressor as SokobanRegressor

RESULTS_DIR = 'surrogate_models/results'
N_FOLDS = 5

all_results = []

print("=== Evaluando CNN (Out-of-Fold) por Bucket ===")
for fold in range(1, N_FOLDS + 1):
    test_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_test.pt")
    weights_path = os.path.join(RESULTS_DIR, f"final_regressor_fold{fold}.pt")
    stats_path = os.path.join(RESULTS_DIR, f"regressor_fold{fold}_stats.pt")
    
    if not os.path.exists(test_path) or not os.path.exists(weights_path):
        continue
        
    print(f"Procesando Fold {fold}...")
    
    # Cargar datos
    test_data = torch.load(test_path, weights_only=False)
    stats = torch.load(stats_path, weights_only=False)
    pushes_mean = stats['pushes_mean']
    pushes_std = stats['pushes_std']
    
    # Configurar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SokobanRegressor().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Inferencia
    # Para evitar OOM, inferir en batches pequeños
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            tensors = torch.stack([item['tensor'] for item in batch]).to(device)
            preds_norm = model(tensors).cpu().numpy()
            
            for j, item in enumerate(batch):
                pred_log = preds_norm[j] * pushes_std + pushes_mean
                pred_raw = np.expm1(pred_log)
                
                all_results.append({
                    'bucket': item['bucket'],
                    'pushes_real': item['pushes_raw'],
                    'cnn_pred': pred_raw
                })

df = pd.DataFrame(all_results)
print(f"\nDataset total evaluado (CNN OOF): {len(df)} ejemplos")

print("\n=== Spearman Intra-Bucket de la CNN ===")
def bucket_sort_key(b):
    if b == '101_plus': return 101
    return int(b.split('_')[0])

results = []
for bucket in sorted(df['bucket'].unique(), key=bucket_sort_key):
    group = df[df['bucket'] == bucket]
    if len(group) < 2: continue
    
    rho, _ = spearmanr(group['cnn_pred'], group['pushes_real'])
    results.append((bucket, len(group), rho))
    print(f"{bucket:<15} | N={len(group):<5} | Spearman={rho:5.3f}")

print("\nPromedio Spearman en buckets difíciles (91+, 101+):")
hard = [r for r in results if r[0] in ('91_to_100', '101_plus')]
if hard:
    avg_rho = np.mean([r[2] for r in hard])
    print(f"  {avg_rho:.3f}")
