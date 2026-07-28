"""
build_difficulty_benchmark.py
─────────────────────────────
Construye un benchmark estratificado por dificultad (pushes óptimos según Hungarian)
con diversidad de cantidad de cajas dentro de cada bucket.

Salidas:
  sok_files/benchmark_stratified.sok  — tableros seleccionados, listos para run_benchmark_stratified.py
  benchmark_stratified_meta.csv       — metadata: board_id, bucket, box_count, source_file

Uso:
  python3 build_difficulty_benchmark.py [--n-per-bucket 5] [--seed 42]
"""

import os
import re
import csv
import random
import argparse
import hashlib
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de buckets (nombre → archivos fuente en sokoban_dataset_buckets/)
# ──────────────────────────────────────────────────────────────────────────────
BUCKETS = {
    "1-10":   ["1_to_10.sok"],
    "11-20":  ["11_to_20.sok"],
    "21-30":  ["21_to_30.sok"],
    "31-50":  ["31_to_40.sok", "41_to_50.sok"],
    "51-70":  ["51_to_60.sok", "61_to_70.sok"],
    "71-90":  ["71_to_80.sok", "81_to_90.sok"],
    "91-100": ["91_to_100.sok"],
    "101+":   ["101_plus.sok"],
}

BUCKET_ORDER = ["1-10", "11-20", "21-30", "31-50", "51-70", "71-90", "91-100", "101+"]
SOK_DIR = "sokoban_dataset_buckets"


def count_boxes(board_str: str) -> int:
    """Cuenta cajas: $ (caja libre) + * (caja sobre meta)."""
    return board_str.count('$') + board_str.count('*')


def board_hash(board_str: str) -> str:
    """Hash SHA-1 corto para identificar tablero único."""
    normalized = '\n'.join(line.rstrip() for line in board_str.strip().split('\n'))
    return hashlib.sha1(normalized.encode()).hexdigest()[:12]


def extract_boards(path: str) -> list[str]:
    """Extrae todos los tableros de un archivo .sok (líneas que empiezan con #)."""
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
    """
    Selecciona n tableros de 'candidates' maximizando diversidad de box_count.
    Estrategia: ordena por box_count, divide en n franjas, toma uno por franja.
    Si hay menos candidatos que n, devuelve todos.
    """
    if len(candidates) <= n:
        return candidates[:]

    # Agrupar por box_count
    by_boxes: dict[int, list[str]] = defaultdict(list)
    for b in candidates:
        by_boxes[count_boxes(b)].append(b)

    box_keys = sorted(by_boxes.keys())

    # Si hay suficientes box_counts distintos, tomar uno de cada franja
    if len(box_keys) >= n:
        # Dividir box_keys en n franjas y tomar uno por franja
        result = []
        step = max(1, len(box_keys) // n)
        for i in range(0, min(n * step, len(box_keys)), step):
            group = by_boxes[box_keys[i]]
            result.append(rng.choice(group))
            if len(result) == n:
                break
        # Si faltan (por redondeo), rellenar con lo que quede
        remaining = [b for b in candidates if b not in result]
        rng.shuffle(remaining)
        result += remaining[:n - len(result)]
        return result[:n]
    else:
        # Pocos box_counts distintos — mezclar y tomar n
        all_b = candidates[:]
        rng.shuffle(all_b)
        return all_b[:n]


def main():
    parser = argparse.ArgumentParser(description="Construye benchmark estratificado por dificultad")
    parser.add_argument('--n-per-bucket', type=int, default=5,
                        help='Tableros a seleccionar por bucket (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para reproducibilidad (default: 42)')
    parser.add_argument('--out-sok', type=str, default='sok_files/benchmark_stratified.sok',
                        help='Archivo .sok de salida')
    parser.add_argument('--out-csv', type=str, default='benchmark_stratified_meta.csv',
                        help='Archivo CSV de metadata de salida')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.n_per_bucket

    print(f"\n{'='*60}")
    print(f" BUILD DIFFICULTY BENCHMARK  (n={n} por bucket, seed={args.seed})")
    print(f"{'='*60}\n")

    # NOTA sobre test/train split:
    # Los archivos en sokoban_dataset_buckets/ contienen el dataset completo
    # (pre-split). Para una separación limpia habría que cruzar los hashes
    # contra dl_dataset_test.pt. Marcamos source_verified_test_split=False
    # en el CSV para que el revisor tenga visibilidad.

    selected_boards = []   # list of (bucket_name, board_str, source_file)
    seen_hashes = set()    # evitar duplicados entre buckets

    for bucket in BUCKET_ORDER:
        files = BUCKETS[bucket]
        candidates = []
        for fname in files:
            path = os.path.join(SOK_DIR, fname)
            if not os.path.exists(path):
                print(f"  [WARN] Archivo no encontrado: {path}")
                continue
            boards = extract_boards(path)
            candidates.extend([(b, fname) for b in boards])

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

        # Recuperar source_file para cada elegido
        chosen_set = {board_hash(b) for b in chosen}
        src_map = {board_hash(b): s for b, s in unique_candidates}

        # Estadísticas para reportar
        box_counts = [count_boxes(b) for b in chosen]
        print(f"  Bucket {bucket:8s}: {len(unique_candidates):5d} candidatos → {len(chosen):2d} seleccionados"
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

    # ── Escribir archivo .sok ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_sok), exist_ok=True)
    with open(args.out_sok, 'w') as f:
        for global_id, entry in enumerate(selected_boards):
            bucket  = entry['bucket']
            boxes   = entry['box_count']
            src     = entry['source_file']
            bstr    = entry['board_str']
            # Header compatible con extract_boards() de run_benchmark.py:
            # la función busca líneas que NO empiezan con '#' como separador,
            # así que simplemente dejamos el tablero y una línea en blanco.
            f.write(f"\n; board_id={global_id}  bucket={bucket}  boxes={boxes}  src={src}\n")
            f.write(bstr)
            f.write("\n\n")

    print(f"\n→ Tableros escritos en:  {args.out_sok}  ({len(selected_boards)} total)")

    # ── Escribir CSV de metadata ─────────────────────────────────────────────
    with open(args.out_csv, 'w', newline='') as csvfile:
        fieldnames = ['board_id', 'bucket', 'box_count', 'source_file', 'board_hash',
                      'source_verified_test_split']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for global_id, entry in enumerate(selected_boards):
            writer.writerow({
                'board_id':                    global_id,
                'bucket':                      entry['bucket'],
                'box_count':                   entry['box_count'],
                'source_file':                 entry['source_file'],
                'board_hash':                  entry['board_hash'],
                'source_verified_test_split':  False,   # ver NOTA arriba
            })

    print(f"→ Metadata escrita en:   {args.out_csv}")
    print(f"\n¡Listo! Corre el benchmark con:")
    print(f"  python3 run_benchmark_stratified.py\n")


if __name__ == '__main__':
    main()
