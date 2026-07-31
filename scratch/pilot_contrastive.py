import os
import subprocess
import time
import sys
import signal

def run_pilot():
    seeds = [str(i) for i in range(42, 52)]
    shells = [f"levels/shell_{i}.sok" for i in range(1, 6)]
    
    print("Starting Surrogate Server... (waiting for logs)")
    server_process = subprocess.Popen(
        ["./venv/bin/python3", "surrogate_models/surrogate_server.py"], 
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    server_ready = False
    while True:
        line = server_process.stdout.readline()
        if not line:
            break
        print("  [SERVER LOG]", line.strip())
        if "Server ready" in line:
            server_ready = True
            break
        if "Running on" in line:
            server_ready = True
            break

    import threading
    def drain_logs(process):
        with open("scratch/surrogate_server.log", "w") as f:
            for line in iter(process.stdout.readline, ''):
                f.write(line)
                f.flush()

    if server_ready:
        threading.Thread(target=drain_logs, args=(server_process,), daemon=True).start()

    if not server_ready:
        print("\n[ERROR] El servidor Surrogate se cerró sin emitir 'Server ready'. Abortando piloto.")
        server_process.kill()
        sys.exit(1)
        
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
                    "--timeLimit", "300",
                    "--maxEvals", "1000000",
                    "--out_csv", "scratch/temp_pilot.csv"
                ]
                is_timeout = False
                try:
                    result = subprocess.run(cmd, timeout=310, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    out_text = result.stdout
                    if result.returncode != 0:
                        print(f"  [ERROR] C++ process CRASHED with returncode {result.returncode}. Full output:\n{out_text}\n{'='*50}")
                except subprocess.TimeoutExpired as e:
                    is_timeout = True
                    # If it times out, the output so far is captured in e.stdout
                    out_text = e.stdout if e.stdout else ""
                    if isinstance(out_text, bytes):
                        out_text = out_text.decode('utf-8', errors='replace')
                    print(f"  [Warning] Seed {seed} timed out by Python after 310s.")
                    print(f"  [DEBUG] Últimas líneas del log de C++ antes de morir:\n{out_text[-2000:]}\n{'='*50}")
                
                disyuntor_count = "TIMEOUT"
                for line in out_text.split("\n"):
                    if "[ES STATS] Circuit Breaker (MAX_FAILURES) triggers:" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            try:
                                disyuntor_count = int(parts[-1].strip())
                            except ValueError:
                                pass
                                
                if disyuntor_count == "TIMEOUT" and not is_timeout:
                    print(f"  [ERROR] C++ process finished but stats not found. Full output:\n{out_text}\n{'='*50}")
                                
                print(f"Shell {shell}, Seed {seed} -> Disyuntor triggers: {disyuntor_count}")
                if isinstance(disyuntor_count, int):
                    total_disyuntor_triggers += disyuntor_count
                    total_runs += 1
                
    finally:
        print("Killing server...")
        server_process.send_signal(signal.SIGINT)
        server_process.wait()
        
    print(f"\nFINAL RESULT: Avg Disyuntor triggers per run = {total_disyuntor_triggers / total_runs:.2f} (Total: {total_disyuntor_triggers})")

if __name__ == "__main__":
    run_pilot()
