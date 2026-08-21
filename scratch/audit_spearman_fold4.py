import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
sys.path.append('surrogate_models')
from models.resnet import SokobanSEResNetRegressor

RESULTS_DIR = "surrogate_models/results"
model_path = os.path.join(RESULTS_DIR, "final_regressor_v2_fold4.pt")
test_data_path = os.path.join(RESULTS_DIR, "regressor_v2_fold4_test.pt")
stats_path = os.path.join(RESULTS_DIR, "regressor_v2_fold4_stats.pt")

def main():
    print("=" * 80)
    print("  AUDITORÍA DE CORRELACIÓN DE RANGO (SPEARMAN) - FOLD 4 (PRODUCCIÓN)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    test_data = torch.load(test_data_path, map_location="cpu", weights_only=False)
    
    print("\n1. Extrayendo predicciones para todos los tableros del Test Set...")
    
    results = []
    with torch.no_grad():
        for item in test_data:
            tensor = item["tensor"].unsqueeze(0).float().to(device)
            real = item["pushes_raw"]
            pred = model(tensor).squeeze(-1).item()
            
            results.append({
                "real": real,
                "pred": pred,
                "bucket": item["bucket"],
                "source": item["source"],
                "shell_hash": item["shell_hash"],
                "tensor_raw": item["tensor"]
            })
            
    df = pd.DataFrame(results)
    
    print("\n" + "-" * 50)
    print("A) SPEARMAN INTRA-BUCKET (Desglose por dificultad)")
    print("-" * 50)
    
    # Ordenar los buckets logicamente
    bucket_order = ["1_to_10", "11_to_20", "21_to_30", "31_to_40", "41_to_50", 
                    "51_to_60", "61_to_70", "71_to_80", "81_to_90", "91_to_100", "101_plus"]
    
    for b in bucket_order:
        b_df = df[df["bucket"] == b]
        if len(b_df) > 1:
            corr, pval = spearmanr(b_df["real"], b_df["pred"])
            print(f"  Bucket {b:>10}: ρ = {corr:+.4f}  (n = {len(b_df):>4})")
    
    print("\n" + "-" * 50)
    print("B) SPEARMAN INTRA-SHELL (Move / Add / Remove)")
    print("-" * 50)
    
    # Agrupar para MOVE (mismo shell, misma cantidad de cajas)
    groups_move = {}
    for r in results:
        t = r["tensor_raw"]
        num_boxes = int(t[2].sum().item())
        key = (r["shell_hash"], num_boxes)
        if key not in groups_move:
            groups_move[key] = []
        groups_move[key].append(r)
        
    valid_move_groups = [g for g in groups_move.values() if len(g) > 1 and len(set([x["real"] for x in g])) > 1]
    
    move_corrs = []
    for g in valid_move_groups:
        real = [x["real"] for x in g]
        pred = [x["pred"] for x in g]
        corr, _ = spearmanr(real, pred)
        if not np.isnan(corr):
            move_corrs.append(corr)
            
    if move_corrs:
        print(f"  Intra-shell MOVE        : ρ = {np.mean(move_corrs):+.4f} (promediado sobre {len(valid_move_groups)} grupos/topologías distintas)")
        
    # Agrupar para ADD/REMOVE (mismo shell, misma posicion de jugador, distinto num_boxes)
    groups_box = {}
    for r in results:
        t = r["tensor_raw"]
        player_pos = tuple((t[4] == 1).nonzero(as_tuple=False).flatten().tolist())
        key = (r["shell_hash"], player_pos)
        if key not in groups_box:
            groups_box[key] = []
        groups_box[key].append(r)
        
    valid_box_groups = [g for g in groups_box.values() if len(g) > 1 and len(set([x["real"] for x in g])) > 1]
    
    box_corrs = []
    for g in valid_box_groups:
        real = [x["real"] for x in g]
        pred = [x["pred"] for x in g]
        corr, _ = spearmanr(real, pred)
        if not np.isnan(corr):
            box_corrs.append(corr)
            
    if box_corrs:
        print(f"  Intra-shell ADD/REMOVE  : ρ = {np.mean(box_corrs):+.4f} (promediado sobre {len(valid_box_groups)} grupos/topologías distintas)")
        
if __name__ == "__main__":
    main()
