import os
import sys
import glob
import json
import hashlib
import torch
import numpy as np
from tqdm import tqdm

# Importar la función canónica de encode_board desde board_utils
# NUNCA reimplementar esta función
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import encode_board

INPUT_DIR = "training_data/SiameseRankNetPilot"
FOLD_MAP_PATH = "surrogate_models/results/fold_map.json"
STATS_PATH = "surrogate_models/results/production_regressor_stats.pt"
OUTPUT_DIR = "surrogate_models/results"

def get_bucket(pushes):
    if pushes <= 10: return "1_to_10"
    if pushes > 100: return "101_plus"
    lower = ((int(pushes) - 1) // 10) * 10 + 1
    upper = lower + 9
    return f"{lower}_to_{upper}"

def main():
    print("="*85)
    print(" 🛠️  PREPARACIÓN DE DATOS: SIAMESE RANKNET INTRA-SHELL PAIRS (MULTI-MACHINE COMBINED)")
    print(" Protocolo sin fugas (Leakage-Free) y calibrado al espacio log1p + z-score")
    print("="*85)

    if not os.path.exists(FOLD_MAP_PATH):
        print(f"❌ Error: No se encontró fold_map.json en {FOLD_MAP_PATH}")
        return

    with open(FOLD_MAP_PATH, "r", encoding="utf-8") as f:
        fold_map = json.load(f)
    print(f"📁 Fold map cargado ({len(fold_map):,} cascarones únicos registrados)")

    if not os.path.exists(STATS_PATH):
        print(f"❌ Error: No se encontraron las estadísticas de normalización en {STATS_PATH}")
        return

    stats = torch.load(STATS_PATH, weights_only=False)
    pushes_mean = stats["pushes_mean"]
    pushes_std = stats["pushes_std"]
    print(f"⚖️  Estadísticas log1p oficiales cargadas: mean={pushes_mean:.4f}, std={pushes_std:.4f}")

    sok_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.sok")))
    if not sok_files:
        print(f"❌ Error: No se encontraron archivos .sok en {INPUT_DIR}")
        return

    print(f"\n📂 Archivos de minería detectados (combinando todas las máquinas):")
    blocks = []
    for fpath in sok_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        f_blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        print(f"   -> {fpath}: {len(f_blocks):,} pares cargados")
        blocks.extend(f_blocks)

    print(f"🔍 Procesando dataset combinado: {len(blocks):,} bloques totales...")

    MOBILE_CHARS = str.maketrans("$.*@+", "     ")
    
    train_val_pairs = []
    test_heldout_pairs = []
    skipped_no_fold = 0
    skipped_parse_error = 0

    for block in tqdm(blocks, desc="Encode Siamese Pairs"):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        try:
            src_idx = lines.index("source_board:")
            mut_idx = lines.index("mutated_board:")
            
            src_lines = lines[src_idx+1 : mut_idx-1]
            src_pushes_line = lines[mut_idx-1]
            src_pushes = float(src_pushes_line.split(":")[1].strip())
            
            mut_lines = lines[mut_idx+1 : len(lines)-1]
            mut_pushes_line = lines[len(lines)-1]
            mut_pushes = float(mut_pushes_line.split(":")[1].strip())

            board_str_A = "\n".join(src_lines)
            board_str_B = "\n".join(mut_lines)

            shell_str = board_str_A.translate(MOBILE_CHARS)
            shell_hash = hashlib.sha256(shell_str.encode("utf-8")).hexdigest()

            if shell_hash not in fold_map:
                skipped_no_fold += 1
                continue

            fold = fold_map[shell_hash]
            
            tensor_A = encode_board(board_str_A)
            tensor_B = encode_board(board_str_B)
            
            norm_A = (np.log1p(src_pushes) - pushes_mean) / (pushes_std + 1e-8)
            norm_B = (np.log1p(mut_pushes) - pushes_mean) / (pushes_std + 1e-8)
            
            item = {
                "tensor_A": torch.tensor(tensor_A, dtype=torch.float32),
                "raw_A": float(src_pushes),
                "norm_A": float(norm_A),
                "bucket_A": get_bucket(src_pushes),
                "tensor_B": torch.tensor(tensor_B, dtype=torch.float32),
                "raw_B": float(mut_pushes),
                "norm_B": float(norm_B),
                "bucket_B": get_bucket(mut_pushes),
                "fold": fold,
                "shell_hash": shell_hash
            }

            if fold == 1:
                test_heldout_pairs.append(item)
            else:
                train_val_pairs.append(item)

        except Exception as e:
            skipped_parse_error += 1
            continue

    print(f"\n✅ Extracción completa:")
    print(f"   -> Train/Val Pairs (Folds 2-5): {len(train_val_pairs):,} pares")
    print(f"   -> Held-Out Test Pairs (Fold 1): {len(test_heldout_pairs):,} pares")
    print(f"   -> Saltados por no estar en fold_map: {skipped_no_fold:,}")
    if skipped_parse_error > 0:
        print(f"   -> Errores de parseo: {skipped_parse_error:,}")

    all_pairs = train_val_pairs + test_heldout_pairs
    if all_pairs:
        diffs = np.array([p["raw_A"] - p["raw_B"] for p in all_pairs])
        abs_diffs = np.abs(diffs)
        
        n_pos = np.sum(diffs > 0) # y_A > y_B
        n_neg = np.sum(diffs < 0) # y_A < y_B
        n_tie = np.sum(diffs == 0) # y_A == y_B

        print("\n" + "="*85)
        print(" 📈 DIAGNÓSTICO DE DISTRIBUCIÓN DEL DATASET COMBINADO (PARA REPORTE A CLAUDE)")
        print("="*85)
        print(f" (1) Distribución de dirección sgn(y_A - y_B):")
        print(f"     -> y_A > y_B (Mutación disminuye empujes) : {n_pos:,} pares ({n_pos/len(all_pairs)*100:.1f}%)")
        print(f"     -> y_A < y_B (Mutación aumenta empujes)   : {n_neg:,} pares ({n_neg/len(all_pairs)*100:.1f}%)")
        if n_tie > 0:
            print(f"     -> y_A == y_B (Empates)                   : {n_tie:,} pares ({n_tie/len(all_pairs)*100:.1f}%)")
        
        print(f"\n (2) Distribución de magnitud |y_A - y_B| (diferencial de empujes reales):")
        print(f"     -> Media: {np.mean(abs_diffs):.2f} | Mediana: {np.median(abs_diffs):.1f} | Desv. Est: {np.std(abs_diffs):.2f}")
        print(f"     -> Mínimo: {np.min(abs_diffs):.1f} | Máximo: {np.max(abs_diffs):.1f}")
        print("     -> Desglose por rango de diferencia:")
        print(f"        * |Δ| <= 1 empuje : {np.sum(abs_diffs <= 1):,} ({np.sum(abs_diffs <= 1)/len(all_pairs)*100:.1f}%)")
        print(f"        * 2 <= |Δ| <= 5    : {np.sum((abs_diffs >= 2) & (abs_diffs <= 5)):,} ({np.sum((abs_diffs >= 2) & (abs_diffs <= 5))/len(all_pairs)*100:.1f}%)")
        print(f"        * 6 <= |Δ| <= 15   : {np.sum((abs_diffs >= 6) & (abs_diffs <= 15)):,} ({np.sum((abs_diffs >= 6) & (abs_diffs <= 15))/len(all_pairs)*100:.1f}%)")
        print(f"        * |Δ| > 15 empujes : {np.sum(abs_diffs > 15):,} ({np.sum(abs_diffs > 15)/len(all_pairs)*100:.1f}%)")
        print("="*85)

    out_train = os.path.join(OUTPUT_DIR, "siamese_ranknet_train.pt")
    out_test  = os.path.join(OUTPUT_DIR, "siamese_ranknet_test_heldout.pt")

    torch.save(train_val_pairs, out_train)
    torch.save(test_heldout_pairs, out_test)
    print(f"\n💾 Archivos guardados en {OUTPUT_DIR}/siamese_ranknet_*.pt")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
