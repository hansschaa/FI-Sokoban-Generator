import os
import subprocess
import time
import signal

def run_pilot():
    seeds = [str(i) for i in range(42, 52)]
    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    
    print("Starting Surrogate Server...")
    server_process = subprocess.Popen(
        ["./venv/bin/python3", "surrogate_models/surrogate_server.py"], 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    while True:
        line = server_process.stdout.readline()
        if not line: break
        if "Server ready" in line: break
        if "Running on" in line: break

    time.sleep(2) # Give it an extra second to bind
    print("Server is up. Running 50 ES Pilot runs...")
    
    total_disyuntor_triggers = 0
    total_runs = 0

    try:
        for shell in shells:
            for seed in seeds:
                runner_path = "./build/experiment_runner"
                if not os.path.exists(runner_path):
                    if os.path.exists("./build2/experiment_runner"):
                        runner_path = "./build2/experiment_runner"
                    else:
                        raise FileNotFoundError("No se encontró experiment_runner en ./build/ ni ./build2/. ¡Compila el código C++ primero!")
                
                cmd = [
                    runner_path, "ES", "FO1", seed, shell,
                    "--heuristic", "neural",
                    "--timeLimit", "120",
                    "--maxEvals", "1000000",
                    "--out_csv", "scratch/temp_pilot.csv"
                ]
                try:
                    result = subprocess.run(cmd, timeout=130, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    out_text = result.stdout
                except subprocess.TimeoutExpired as e:
                    # If it times out, the output so far is captured in e.stdout
                    out_text = e.stdout if e.stdout else ""
                    print(f"  [Warning] Seed {seed} timed out by Python after 130s.")
                
                disyuntor_count = 0
                for line in out_text.split("\n"):
                    if "Disyuntor" in line or "circuit breaker" in line.lower() or "clones" in line.lower() or "diversity" in line.lower():
                        if "triggered" in line.lower() or "activated" in line.lower() or "[DIVERSITY]" in line:
                            if "[DIVERSITY]" in line:
                                disyuntor_count += 1
                                
                print(f"Shell {shell}, Seed {seed} -> Disyuntor triggers: {disyuntor_count}")
                total_disyuntor_triggers += disyuntor_count
                total_runs += 1
                
    finally:
        print("Killing server...")
        server_process.send_signal(signal.SIGINT)
        server_process.wait()
        
    print(f"\nFINAL RESULT: Avg Disyuntor triggers per run = {total_disyuntor_triggers / total_runs:.2f} (Total: {total_disyuntor_triggers})")

if __name__ == "__main__":
    run_pilot()
