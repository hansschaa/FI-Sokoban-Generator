import subprocess

def main():
    cmd = [
        "./build/experiment_runner", "ES", "FO1", "42", "levels/shell_1.sok",
        "--heuristic", "full_surrogate",
        "--timeLimit", "300",
        "--maxEvals", "1000",
        "--mu", "9",
        "--lambda", "28",
        "--mutRate", "0.8559",
        "--stagLimit", "5"  # Solo correr unas pocas generaciones para ver el log
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    print("=== RAW TIMING LOG (SEED 42, FULL SURROGATE, PARALLEL) ===")
    lines = res.stdout.split('\n')
    for line in lines:
        if "[ES_TIMING]" in line or "[TIMING_PHASE]" in line:
            print(line)

if __name__ == "__main__":
    main()
