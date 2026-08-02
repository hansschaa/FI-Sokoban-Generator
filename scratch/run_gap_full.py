import os
import subprocess
import json
import statistics
import sys

def main():
    alg = sys.argv[1] if len(sys.argv) > 1 else "GA"
    
    shells = [
        "levels/centroid_shell_1.sok",
        "levels/centroid_shell_2.sok",
        "levels/centroid_shell_3.sok",
        "levels/centroid_shell_4.sok",
        "levels/centroid_shell_5.sok",
    ]
    
    binary_path = "./build/experiment_runner"
    solver_path = "./build/test_solver"
    
    print("========================================")
    print(f" RUNNING FULL GAP DECOMPOSITION ({alg} - 15 SEEDS)")
    print("========================================")
    
    # Optional: write to CSV
    csv_file = f"scratch/full_gap_results_{alg}.csv"
    with open(csv_file, "w") as f:
        f.write("Shell,Seed,PredictedPushes,TruePushes,Gap\n")
        
    shell_gaps = {s: [] for s in shells}
    
    for i, shell in enumerate(shells):
        print(f"\n[{i+1}/5] Running {alg} on {shell} (15 Seeds)...")
        
        for seed in range(1, 16):
            cmd_ga = [binary_path, alg, "FO1", str(seed), shell, "--heuristic", "full_surrogate", "--maxEvals", "100000"]
            res_ga = subprocess.run(cmd_ga, stdout=subprocess.PIPE, text=True)
            
            lines = [line for line in res_ga.stdout.strip().split('\n') if ';' in line]
            if not lines:
                print(f"  [Seed {seed}] Error: No valid output from GA.")
                continue
                
            last_line = lines[-1]
            output = last_line.strip().split(";")
            predicted_fitness = abs(float(output[0]))
            generated_board = output[2]
            
            temp_board = "scratch/temp_full_board.txt"
            with open(temp_board, "w") as f:
                f.write(generated_board.replace('|', '\n').strip())
                
            cmd_astar = [solver_path, temp_board, "hungarian"]
            res_astar = subprocess.run(cmd_astar, stdout=subprocess.PIPE, text=True)
            
            true_pushes = -1
            for line in res_astar.stdout.split('\n'):
                if line.startswith("pushes:"):
                    true_pushes = int(line.split(":")[1].strip())
                    
            gap = abs(predicted_fitness - true_pushes)
            shell_gaps[shell].append(gap)
            
            with open(csv_file, "a") as f:
                f.write(f"{shell},{seed},{predicted_fitness:.1f},{true_pushes},{gap:.1f}\n")
                
            print(f"  -> Seed {seed:02d} | Pred: {predicted_fitness:5.1f} | True: {true_pushes:5d} | Gap: {gap:5.1f}")
            
        avg_gap = statistics.mean(shell_gaps[shell]) if shell_gaps[shell] else 0.0
        print(f"==> AVERAGE GAP FOR {os.path.basename(shell)}: {avg_gap:.2f}")

    print("\n========================================")
    print(" FINAL SUMMARY (AVERAGE GAPS)")
    print("========================================")
    for i, shell in enumerate(shells):
        avg = statistics.mean(shell_gaps[shell]) if shell_gaps[shell] else 0.0
        print(f"Shell {i+1}: {avg:.2f}")

    print("\n========================================")
    print(" CHECKING CALIBRATION COUNTERS")
    print("========================================")
    stats_file = "scratch/calibration_stats.json"
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)
        print(f"Out of Bounds Count: {stats.get('oob_count', 0)}")
        print(f"Clip Override Count: {stats.get('clip_override_count', 0)}")
    else:
        print("No calibration stats found.")

if __name__ == "__main__":
    main()
