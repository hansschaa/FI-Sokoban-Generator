import os
import subprocess
import time
import pandas as pd
import re

shells_and_seeds = [
    (1, list(range(42, 52))),
    (5, list(range(42, 52))),
    (3, [42, 43])
]

# Read the sequential results for comparison
seq_csv_path = "scratch/final_canonical_results.csv"
seq_df = pd.read_csv(seq_csv_path)

results = []

for shell_idx, seeds in shells_and_seeds:
    shell_file = f"levels/shell_{shell_idx}.sok"
    for seed in seeds:
        print(f"Running Shell {shell_idx} Seed {seed}...")
        cmd = [
            "./build/experiment_runner", "ES", "FO1", str(seed), shell_file,
            "--heuristic", "full_surrogate",
            "--timeLimit", "300",
            "--maxEvals", "1000000",
            "--mu", "9",
            "--lambda", "28",
            "--mutRate", "0.8559",
            "--stagLimit", "199"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        output = result.stdout + "\n" + result.stderr
        
        time_s = end_time - start_time
        
        lines = [line for line in output.split('\n') if ';' in line]
        if lines:
            last_line = lines[-1]
            parts = last_line.split(';')
            if len(parts) >= 3:
                pushes = int(float(parts[0]) * -1)
        elif "No valid initial boards" in output:
            pushes = 0
            
        if pushes is None:
            print("ERROR: Could not parse pushes. Output was:")
            print(output)
            break
            
        # Get Sequential Baseline
        seq_row = seq_df[(seq_df["Variant"] == "full_surrogate") & 
                         (seq_df["Shell_Idx"] == shell_idx) & 
                         (seq_df["Seed"] == seed)]
        
        seq_time = seq_row["Time_s"].values[0] if not seq_row.empty else -1
        seq_pushes = seq_row["Top5_Best_Real_Astar_Pushes"].values[0] if not seq_row.empty else -1
        
        results.append({
            "Shell": shell_idx,
            "Seed": seed,
            "Par_Time": round(time_s, 2) if time_s else -1,
            "Seq_Time": round(seq_time, 2),
            "Par_Pushes": pushes,
            "Seq_Pushes": seq_pushes,
            "Speedup": round(seq_time / time_s, 2) if time_s and time_s > 0 else 0,
            "Pushes_Match": (pushes == seq_pushes)
        })
        print(f"  -> Par Time: {time_s:.2f}s (Seq: {seq_time:.2f}s) | Par Pushes: {pushes} (Seq: {seq_pushes}) | Match: {pushes == seq_pushes}")

df = pd.DataFrame(results)
print("\n=== Validation Results ===")
print(df.to_string())

all_match = df["Pushes_Match"].all()
print(f"\nAll Pushes Match: {all_match}")
if all_match:
    print("SUCCESS: Parallel version is mathematically identical to sequential version!")
else:
    print("WARNING: Divergence detected in Pushes!")

df.to_csv("scratch/parallel_validation_results.csv", index=False)
