import subprocess
import os
import time
import signal
import multiprocessing

print("=================================================================")
print(" EXPERIMENT HARDWARE & PARALLELISM LOGGING")
print("=================================================================")
print(f"CPU Cores Available for Hungarian (A*): {multiprocessing.cpu_count()} threads (via std::async)")
print("GPU for Surrogate Model: Information will be logged by the server")
print("=================================================================")

print("Starting Surrogate Server...")
server_process = subprocess.Popen(["./venv/bin/python3", "surrogate_models/surrogate_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait for server to be ready and print hashes
print("Waiting for server to load models...")
while True:
    line = server_process.stdout.readline()
    if not line:
        break
    print(f"[Server] {line.strip()}")
    if "Server ready" in line:
        break

print("Server is ready. Starting experiment script...")
# Now run ablation script
try:
    subprocess.run(["./venv/bin/python3", "surrogate_models/ablation_metaheuristics.py"], check=True)
except Exception as e:
    print(f"Experiment failed: {e}")
finally:
    print("Killing server...")
    server_process.send_signal(signal.SIGINT)
    server_process.wait()
    print("Done.")
