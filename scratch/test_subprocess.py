import subprocess
import os

env = os.environ.copy()
cmd = [
    "./build/experiment_runner",
    "ES",
    "FO1",
    "90",
    "levels/shell_1.sok",
    "--heuristic", "neural",
    "--timeLimit", "15",
    "--maxEvals", "1000000",
    "--out_csv", "optuna_results/test90.tmp"
]

res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("RETURN CODE:", res.returncode)
print("STDOUT LENGTH:", len(res.stdout))
print("CONTAINS DIVERSITY:", "[DIVERSITY]" in res.stdout)
for line in res.stdout.split('\n'):
    if "[DIVERSITY]" in line:
        print(line.strip())
