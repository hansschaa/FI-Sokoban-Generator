import csv
from collections import defaultdict
import sys
import pandas as pd

def intersection_summary(csv_file):
    board_status = defaultdict(dict)
    board_nodes = defaultdict(dict)
    board_times = defaultdict(dict)
    board_pushes = defaultdict(dict)
    
    heuristics = set()
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                heur = row["heuristic"]
                board_id = int(row["board_id"])
                status = row["status"]
                
                heuristics.add(heur)
                board_status[board_id][heur] = status
                if status == "SOLVED":
                    board_nodes[board_id][heur] = int(row["expanded_nodes"])
                    board_times[board_id][heur] = float(row["runtime_ms"])
                    board_pushes[board_id][heur] = int(row["pushes"])
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

    # Filter only heuristics we care about for the intersection
    # Assuming we want intersection of manhattan, hungarian, neural_sequential, neural_batched, neural_batched_massive
    core_heuristics = [h for h in ["manhattan", "hungarian", "neural_sequential", "neural_batched", "neural_batched_massive"] if h in heuristics]
    if not core_heuristics:
        print("No valid heuristics found.")
        return

    print("=== Análisis de Tableros y Fallos ===")
    for board_id, statuses in sorted(board_status.items()):
        failed = []
        for h in core_heuristics:
            if statuses.get(h) != "SOLVED":
                failed.append(f"{h} ({statuses.get(h, 'MISSING')})")
        if failed:
            print(f"Tablero {board_id}: Falló en {', '.join(failed)}")

    intersection_boards = []
    for board_id, statuses in sorted(board_status.items()):
        if all(statuses.get(h) == "SOLVED" for h in core_heuristics):
            intersection_boards.append(board_id)
            
    print(f"\nTableros resueltos por TODAS las {len(core_heuristics)} heurísticas centrales: {len(intersection_boards)}")
    
    if not intersection_boards:
        return
        
    print("\n=== Tabla Detallada (Solo Intersección) ===")
    
    data = []
    for board_id in intersection_boards:
        row = {"board_id": board_id}
        for h in core_heuristics:
            row[f"{h}_nodes"] = board_nodes[board_id][h]
            row[f"{h}_time"] = board_times[board_id][h]
            row[f"{h}_pushes"] = board_pushes[board_id][h]
            
        # Pushes diff neural_sequential - hungarian
        if "neural_sequential" in core_heuristics and "hungarian" in core_heuristics:
            row["push_diff_seq_vs_hungarian"] = board_pushes[board_id]["neural_sequential"] - board_pushes[board_id]["hungarian"]
            
        data.append(row)
        
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    
    print("\n=== Promedios Agregados (Intersección) ===")
    stats = defaultdict(lambda: {"nodes_sum": 0, "time_sum": 0, "pushes_sum": 0})
    for board_id in intersection_boards:
        for h in core_heuristics:
            stats[h]["nodes_sum"] += board_nodes[board_id][h]
            stats[h]["time_sum"] += board_times[board_id][h]
            stats[h]["pushes_sum"] += board_pushes[board_id][h]
            
    print(f"{'Heuristic':<25} | {'Avg Nodes':<15} | {'Avg Time (ms)':<15} | {'Avg Pushes':<15}")
    print("-" * 75)
    for h in core_heuristics:
        avg_nodes = stats[h]["nodes_sum"] / len(intersection_boards)
        avg_time = stats[h]["time_sum"] / len(intersection_boards)
        avg_pushes = stats[h]["pushes_sum"] / len(intersection_boards)
        print(f"{h:<25} | {avg_nodes:<15.1f} | {avg_time:<15.1f} | {avg_pushes:<15.1f}")
        
    if "neural_sequential" in core_heuristics and "hungarian" in core_heuristics:
        ratio_nodes = stats["neural_sequential"]["nodes_sum"] / stats["hungarian"]["nodes_sum"]
        print(f"\nRatio Global de Nodos (neural_sequential / hungarian): {ratio_nodes:.3f}x")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 intersection_benchmark.py <archivo_csv>")
    else:
        intersection_summary(sys.argv[1])
