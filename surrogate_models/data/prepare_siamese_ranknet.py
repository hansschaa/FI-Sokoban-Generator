import os
import sys
import json
import hashlib
import torch
import numpy as np
from tqdm import tqdm

# Importar la función canónica de encode_board desde board_utils
# NUNCA reimplementar esta función
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_utils import encode_board

INPUT_SOK = "training_data/SiameseRankNetPilot/ranknet_intra_shell_pairs.sok"
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
    print("="*80)
    print(" 🛠️  PREPARACIÓN DE DATOS: SIAMESE RANKNET INTRA-SHELL PAIRS")
    print(" Protocolo sin fugas (Leakage-Free) y calibrado al espacio log1p + z-score")
    print("="*80)

    if not os.path.exists(INPUT_SOK):
        print(f"❌ Error: No se encontró el archivo de pares en {INPUT_SOK}")
        return

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

    with open(INPUT_SOK, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    print(f"🔍 Procesando {len(blocks):,} bloques de pares...")

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

    out_train = os.path.join(OUTPUT_DIR, "siamese_ranknet_train.pt")
    out_test  = os.path.join(OUTPUT_DIR, "siamese_ranknet_test_heldout.pt")

    torch.save(train_val_pairs, out_train)
    torch.save(test_heldout_pairs, out_test)
    print(f"\n💾 Archivos guardados en {OUTPUT_DIR}/siamese_ranknet_*.pt")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
