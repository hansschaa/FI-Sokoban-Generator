import csv
from collections import defaultdict
import sys

def summarize(csv_file):
    stats = defaultdict(lambda: {"total": 0, "solved": 0, "nodes_sum": 0, "time_sum": 0})
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                heur = row["heuristic"]
                status = row["status"]
                
                stats[heur]["total"] += 1
                if status == "SOLVED":
                    stats[heur]["solved"] += 1
                    stats[heur]["nodes_sum"] += int(row["expanded_nodes"])
                    stats[heur]["time_sum"] += float(row["runtime_ms"])
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return

    print(f"{'Heuristic':<25} | {'Solved / Total':<15} | {'% Solved':<10} | {'Avg Nodes':<10} | {'Avg Time (ms)':<15}")
    print("-" * 85)
    for heur, s in stats.items():
        total = s["total"]
        solved = s["solved"]
        perc = (solved / total * 100) if total > 0 else 0
        avg_nodes = (s["nodes_sum"] / solved) if solved > 0 else 0
        avg_time = (s["time_sum"] / solved) if solved > 0 else 0
        print(f"{heur:<25} | {solved:>6} / {total:<6} | {perc:>8.1f}% | {avg_nodes:>10.1f} | {avg_time:>13.1f}")

if __name__ == "__main__":
    summarize(sys.argv[1])
