import csv
from collections import defaultdict
import sys

def intersection_summary(csv_file):
    board_status = defaultdict(dict)
    board_nodes = defaultdict(dict)
    board_times = defaultdict(dict)
    
    heuristics = set()
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                heur = row["heuristic"]
                board_id = row["board_id"]
                status = row["status"]
                
                # Only care about these 3 heuristics
                if heur not in ["manhattan", "hungarian", "neural_batched_massive"]:
                    continue
                    
                heuristics.add(heur)
                board_status[board_id][heur] = status
                if status == "SOLVED":
                    board_nodes[board_id][heur] = int(row["expanded_nodes"])
                    board_times[board_id][heur] = float(row["runtime_ms"])
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

    # 1. Analizar fallos
    print("=== Análisis de Tableros y Fallos ===")
    for board_id, statuses in board_status.items():
        failed = []
        for h in heuristics:
            if statuses.get(h) != "SOLVED":
                failed.append(f"{h} ({statuses.get(h, 'MISSING')})")
        if failed:
            print(f"Tablero {board_id}: Falló en {', '.join(failed)}")

    # 2. Computar intersección
    intersection_boards = []
    for board_id, statuses in board_status.items():
        if all(statuses.get(h) == "SOLVED" for h in heuristics):
            intersection_boards.append(board_id)
            
    print(f"\nTableros resueltos por TODAS las {len(heuristics)} heurísticas: {len(intersection_boards)}")
    
    # 3. Calcular promedios en la intersección
    stats = defaultdict(lambda: {"nodes_sum": 0, "time_sum": 0})
    for board_id in intersection_boards:
        for h in heuristics:
            stats[h]["nodes_sum"] += board_nodes[board_id][h]
            stats[h]["time_sum"] += board_times[board_id][h]
            
    print(f"\n{'Heuristic':<25} | {'Avg Nodes':<15} | {'Avg Time (ms)':<15}")
    print("-" * 60)
    for h in sorted(list(heuristics)):
        avg_nodes = stats[h]["nodes_sum"] / len(intersection_boards) if intersection_boards else 0
        avg_time = stats[h]["time_sum"] / len(intersection_boards) if intersection_boards else 0
        print(f"{h:<25} | {avg_nodes:<15.1f} | {avg_time:<15.1f}")

if __name__ == "__main__":
    intersection_summary(sys.argv[1])
