import os
import subprocess
import json

def main():
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
    print(" RUNNING GAP PILOT ON 5 CENTROID SHELLS")
    print("========================================")
    
    for i, shell in enumerate(shells):
        print(f"\n[{i+1}/5] Running GA on {shell}...")
        
        # Run GA
        cmd_ga = [binary_path, "GA", "FO1", "1", shell, "--heuristic", "full_surrogate", "--maxEvals", "10000"]
        res_ga = subprocess.run(cmd_ga, stdout=subprocess.PIPE, text=True)
        
        lines = [line for line in res_ga.stdout.strip().split('\n') if ';' in line]
        if not lines:
            print(f"  Error: No valid output from GA. Output:\n{res_ga.stdout}")
            continue
            
        last_line = lines[-1]
        output = last_line.strip().split(";")
        predicted_fitness = abs(float(output[0]))
        generated_board = output[2]
        
        # Save board to temp
        temp_board = "scratch/temp_pilot_board.txt"
        with open(temp_board, "w") as f:
            f.write(generated_board.replace('|', '\n').strip())
            
        # Run A* verifier
        cmd_astar = [solver_path, temp_board, "hungarian"]
        res_astar = subprocess.run(cmd_astar, stdout=subprocess.PIPE, text=True)
        
        true_pushes = -1
        for line in res_astar.stdout.split('\n'):
            if line.startswith("pushes:"):
                true_pushes = int(line.split(":")[1].strip())
                
        gap = predicted_fitness - true_pushes
        print(f"  -> Predicted Pushes: {predicted_fitness:.1f}")
        print(f"  -> True A* Pushes:   {true_pushes}")
        print(f"  -> Gap (Pred - True): {gap:.1f}")

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
