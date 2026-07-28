"""
build_fold_heldout_benchmark.py
─────────────────────────────
Construye un benchmark estratificado por dificultad utilizando EXCLUSIVAMENTE
tableros que pertenecen al Test Set (held-out) de un Fold específico de CV.

Salidas:
  sok_files/benchmark_stratified_heldout.sok
  benchmark_stratified_heldout_meta.csv

Uso:
  python3 build_fold_heldout_benchmark.py [--fold 2] [--n-per-bucket 5] [--seed 42]
"""

import os
import csv
import random
import argparse
import hashlib
from collections import defaultdict
import torch

SOK_DIR = "sokoban_dataset_buckets"
BUCKET_ORDER = ["1_to_10", "11_to_20", "21_to_30", "31_to_40", "41_to_50", "51_to_60", "61_to_70", "71_to_80", "81_to_90", "91_to_100", "101_plus"]

# Mapeo de nombres limpios a los nombres en el pt
BUCKET_CLEAN = {
    "1_to_10":   "1-10",
    "11_to_20":  "11-20",
    "21_to_30":  "21-30",
    "31_to_40":  "31-50",
    "41_to_50":  "31-50",
    "51_to_60":  "51-70",
    "61_to_70":  "51-70",
    "71_to_80":  "71-90",
    "81_to_90":  "71-90",
    "91_to_100": "91-100",
    "101_plus":  "101+",
}

# Para la metaestructura, agruparemos usando BUCKET_ORDER_CLEAN
BUCKET_ORDER_CLEAN = ["1-10", "11-20", "21-30", "31-50", "51-70", "71-90", "91-100", "101+"]

def count_boxes(board_str: str) -> int:
    return board_str.count('$') + board_str.count('*')

def board_hash(board_str: str) -> str:
    normalized = '\n'.join(line.rstrip() for line in board_str.strip().split('\n'))
    return hashlib.sha1(normalized.encode()).hexdigest()[:12]

def get_shell_hash(board_str: str) -> str:
    # La misma lógica que prepare_regressor.py
    MOBILE_CHARS = str.maketrans("$.*@+", "     ")
    shell_str = board_str.translate(MOBILE_CHARS)
    return hashlib.sha256(shell_str.encode()).hexdigest()

def extract_boards(path: str) -> list[str]:
    with open(path) as f:
        content = f.read()
    boards, cur = [], []
    for line in content.split('\n'):
        if line.strip().startswith('#'):
            cur.append(line.rstrip())
        elif cur:
            boards.append('\n'.join(cur))
            cur = []
    if cur:
        boards.append('\n'.join(cur))
    return boards

def select_diverse(candidates: list[str], n: int, rng: random.Random) -> list[str]:
    if len(candidates) <= n:
        return candidates[:]
    by_boxes: dict[int, list[str]] = defaultdict(list)
    for b in candidates:
        by_boxes[count_boxes(b)].append(b)
    box_keys = sorted(by_boxes.keys())
    if len(box_keys) >= n:
        result = []
        step = max(1, len(box_keys) // n)
        for i in range(0, min(n * step, len(box_keys)), step):
            group = by_boxes[box_keys[i]]
            result.append(rng.choice(group))
            if len(result) == n:
                break
        remaining = [b for b in candidates if b not in result]
        rng.shuffle(remaining)
        result += remaining[:n - len(result)]
        return result[:n]
    else:
        all_b = candidates[:]
        rng.shuffle(all_b)
        return all_b[:n]

def main():
    parser = argparse.ArgumentParser(description="Construye benchmark estratificado de TEST HELD-OUT puro")
    parser.add_argument('--fold', type=int, default=2, help='Fold cuyo Test Set se usará')
    parser.add_argument('--n-per-bucket', type=int, default=5, help='Tableros por bucket')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--out-sok', type=str, default='sok_files/benchmark_stratified_heldout.sok')
    parser.add_argument('--out-csv', type=str, default='benchmark_stratified_heldout_meta.csv')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.n_per_bucket

    print(f"\n{'='*70}")
    print(f" BUILD HELDOUT BENCHMARK (Fold {args.fold}, n={n} por bucket, seed={args.seed})")
    print(f"{'='*70}\n")

    # 1. Cargar el Test Set del Fold
    pt_path = f"surrogate_models/results/regressor_fold{args.fold}_test.pt"
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"No se encontró {pt_path}")
    
    print(f"Cargando hashes del test set de Fold {args.fold}...")
    test_data = torch.load(pt_path, weights_only=False)
    test_shell_hashes = {d['shell_hash'] for d in test_data}
    print(f"  Encontrados {len(test_shell_hashes)} cascarones en el test set de Fold {args.fold}.")

    # 2. Cargar todos los tableros del dataset y mapear
    all_candidates_by_bucket = defaultdict(list)
    print("Mapeando tableros originales y cruzando con el test set...")
    for fname in os.listdir(SOK_DIR):
        if not fname.endswith('.sok'): continue
        path = os.path.join(SOK_DIR, fname)
        boards = extract_boards(path)
        for b in boards:
            sh = get_shell_hash(b)
            if sh in test_shell_hashes:
                # Determinar bucket basado en el nombre del archivo
                raw_bucket = fname.replace('.sok', '')
                clean_bucket = BUCKET_CLEAN.get(raw_bucket)
                if clean_bucket:
                    all_candidates_by_bucket[clean_bucket].append((b, fname))

    selected_boards = []
    seen_hashes = set()

    for bucket in BUCKET_ORDER_CLEAN:
        candidates = all_candidates_by_bucket[bucket]
        
        # Deduplicar
        unique_candidates = []
        for board_str, src in candidates:
            h = board_hash(board_str)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_candidates.append((board_str, src))

        rng.shuffle(unique_candidates)
        boards_only = [b for b, _ in unique_candidates]
        chosen = select_diverse(boards_only, n, rng)

        src_map = {board_hash(b): s for b, s in unique_candidates}
        box_counts = [count_boxes(b) for b in chosen]
        
        print(f"  Bucket {bucket:8s}: {len(unique_candidates):5d} candidatos held-out → {len(chosen):2d} seleccionados"
              f"  boxes={sorted(box_counts)}")

        for board_str in chosen:
            h = board_hash(board_str)
            selected_boards.append({
                'bucket': bucket,
                'board_str': board_str,
                'source_file': src_map.get(h, 'unknown'),
                'box_count': count_boxes(board_str),
                'board_hash': h,
            })

    os.makedirs(os.path.dirname(args.out_sok), exist_ok=True)
    with open(args.out_sok, 'w') as f:
        for global_id, entry in enumerate(selected_boards):
            f.write(f"\n; board_id={global_id}  bucket={entry['bucket']}  boxes={entry['box_count']}  src={entry['source_file']}\n")
            f.write(entry['board_str'])
            f.write("\n\n")
    print(f"\n→ Tableros escritos en:  {args.out_sok}  ({len(selected_boards)} total)")

    with open(args.out_csv, 'w', newline='') as csvfile:
        fieldnames = ['board_id', 'bucket', 'box_count', 'source_file', 'board_hash', 'source_verified_test_split', 'eval_fold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for global_id, entry in enumerate(selected_boards):
            writer.writerow({
                'board_id':                    global_id,
                'bucket':                      entry['bucket'],
                'box_count':                   entry['box_count'],
                'source_file':                 entry['source_file'],
                'board_hash':                  entry['board_hash'],
                'source_verified_test_split':  True,
                'eval_fold':                   args.fold,
            })
    print(f"→ Metadata escrita en:   {args.out_csv}")
    print("\n¡Listo! Ahora corre:")
    print(f"  python3 run_benchmark_stratified.py --meta {args.out_csv} --sok {args.out_sok} --out benchmark_heldout_results.csv")

if __name__ == '__main__':
    main()
