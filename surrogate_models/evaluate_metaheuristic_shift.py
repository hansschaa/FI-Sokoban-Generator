import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from data.board_utils import encode_board
from models.resnet import SokobanSEResNetRegressor

def main():
    parser = argparse.ArgumentParser(description="Evaluar Regresor en tableros generados por GA/ES (Distribution Shift)")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV con tableros de la metaheurística (debe tener 'board_str' y 'pushes')")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado (ej. final_regressor_fold1.pt)")
    parser.add_argument("--stats", type=str, required=True, help="Ruta a los stats de normalización (ej. regressor_fold1_stats.pt)")
    args = parser.parse_args()

    print(f"Cargando dataset: {args.csv}")
    df = pd.read_csv(args.csv)
    if "board_str" not in df.columns or "pushes" not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'board_str' y 'pushes'")

    print(f"Cargando estadísticas desde: {args.stats}")
    stats = torch.load(args.stats, weights_only=False)
    p_mean, p_std = stats["pushes_mean"], stats["pushes_std"]

    print(f"Cargando modelo desde: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    all_p_pred = []
    all_p_raw = []
    
    print("Evaluando tableros...")
    with torch.no_grad():
        for _, row in df.iterrows():
            board_str = row["board_str"]
            p_raw = float(row["pushes"])
            
            t = encode_board(board_str)
            tensor = torch.tensor(t).unsqueeze(0).float().to(device)
            
            p_pred = model(tensor)
            p_desnorm = p_pred.cpu().item() * p_std + p_mean
            p_desnorm_real = np.expm1(p_desnorm)
            
            all_p_pred.append(p_desnorm_real)
            all_p_raw.append(p_raw)

    all_p_pred = np.array(all_p_pred)
    all_p_raw = np.array(all_p_raw)

    mae = np.mean(np.abs(all_p_pred - all_p_raw))
    spearman_rho, _ = spearmanr(all_p_raw, all_p_pred)

    print("=" * 50)
    print(" RESULTADOS DE DISTRIBUTION SHIFT")
    print("=" * 50)
    print(f" Total de tableros evaluados : {len(df)}")
    print(f" MAE (Error Absoluto Medio)  : {mae:.2f} empujes")
    print(f" Correlación Spearman ρ      : {spearman_rho:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
