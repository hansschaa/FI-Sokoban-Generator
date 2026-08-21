"""
verify_regressor_v2.py
--------------------
Evalúa un regresor entrenado sobre los test sets generados por prepare_regressor_v2.py.
Desglosa el Error Absoluto Medio (MAE) y el Error Cuadrático Medio (MSE) en espacio
real de Pushes, separando los resultados por origen (Denso vs Original) para confirmar
la generalización.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.resnet import SokobanSEResNetRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main():
    print("=" * 70)
    print("  VERIFICACIÓN DE GENERALIZACIÓN DEL REGRESOR (Denso vs Original)")
    print("=" * 70)
    
    # Load model (can be fold1 or the final production one)
    # Por defecto probamos el de fold 1 para la validacion inicial
    model_path = os.path.join(RESULTS_DIR, "final_regressor_v2_fold1.pt")
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo: {model_path}")
        print("Asegurate de haber entrenado al menos el fold 1 primero.")
        return

    print(f"Cargando modelo: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SokobanSEResNetRegressor(dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_path = os.path.join(RESULTS_DIR, "regressor_v2_fold1_test.pt")
    stats_path = os.path.join(RESULTS_DIR, "regressor_v2_fold1_stats.pt")
    
    if not os.path.exists(test_path):
        print(f"❌ No se encontró el test set: {test_path}")
        return

    print(f"Cargando conjunto de test: {test_path}")
    test_data = torch.load(test_path, map_location="cpu")
    stats = torch.load(stats_path, map_location="cpu")
    pushes_mean = stats["pushes_mean"]
    pushes_std = stats["pushes_std"]

    results = []
    
    print("\nEvaluando tensores...")
    with torch.no_grad():
        for item in test_data:
            tensor = item["tensor"].unsqueeze(0).float().to(device)
            real_pushes = item["pushes_raw"]
            source = item["source"]
            bucket = item["bucket"]

            pred_norm = model(tensor).item()
            pred_log = (pred_norm * pushes_std) + pushes_mean
            pred_pushes = np.expm1(pred_log)
            
            abs_err = abs(pred_pushes - real_pushes)
            sq_err = (pred_pushes - real_pushes) ** 2

            results.append({
                "source": source,
                "bucket": bucket,
                "real": real_pushes,
                "pred": pred_pushes,
                "mae": abs_err,
                "mse": sq_err
            })

    df = pd.DataFrame(results)
    
    print("\n[!] Resultados Generales por Origen (Denso vs Original):")
    summary = df.groupby("source").agg(
        Count=("real", "count"),
        MAE=("mae", "mean"),
        MSE=("mse", "mean")
    ).reset_index().round(2)
    print(summary.to_string(index=False))
    
    print("\n[!] Resultados Detallados (Origen x Bucket):")
    detailed = df.groupby(["source", "bucket"]).agg(
        Count=("real", "count"),
        MAE=("mae", "mean")
    ).reset_index().round(2)
    print(detailed.to_string(index=False))
    
    # Comprobar la brecha
    try:
        mae_original = summary[summary["source"] == "Original"]["MAE"].values[0]
        mae_denso = summary[summary["source"] == "Denso"]["MAE"].values[0]
        brecha = abs(mae_original - mae_denso)
        print(f"\n🔬 Brecha de MAE entre dominios: {brecha:.2f} pushes")
        if brecha < 5.0:
            print("✅ EXCELENTE: El regresor generaliza de forma balanceada a la topología densa.")
        else:
            print("⚠️ ATENCIÓN: Todavía hay una brecha significativa en el error entre dominios.")
    except:
        pass

if __name__ == "__main__":
    main()
