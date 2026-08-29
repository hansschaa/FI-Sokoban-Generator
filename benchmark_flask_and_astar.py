import time
import json
import requests
import torch
import subprocess
import os
import sys

# Agregar surrogate_models al sys.path para que los imports internos funcionen
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surrogate_models'))

from surrogate_models.surrogate_server import get_hungarian_lb
from data.prepare_classifier import encode_board

# Tablero gigante típico de Shell 5 (13x13, 5 cajas)
board_shell5 = """
#############
#           #
# .$.$.$.$. #
#.$@$$$$$$$.#
# .$.$.$.$. #
#           #
#############
"""

batch_size = 50
payload = {"boards": [{"board": board_shell5, "parent_board": board_shell5}] * batch_size}

print(f"=== INICIANDO BENCHMARK aislando cuellos de botella (Shell 5) ===")

# 1. Testear Flask completo (End-to-End)
try:
    t0 = time.time()
    res = requests.post("http://127.0.0.1:5000/evaluate", json=payload)
    t1 = time.time()
    if res.status_code == 200:
        print(f"[1] Flask Completo (Red+BFS): {(t1-t0)*1000:.2f} ms total -> {((t1-t0)*1000)/batch_size:.2f} ms/eval")
    else:
        print(f"[1] Flask falló con código {res.status_code}")
except Exception as e:
    print(f"[1] Error conectando a Flask: {e}")

# 2. Testear ÚNICAMENTE el BFS Python (get_hungarian_lb)
tensor = torch.from_numpy(encode_board(board_shell5))
t2 = time.time()
for _ in range(batch_size):
    get_hungarian_lb(tensor)
t3 = time.time()
print(f"[2] Solo BFS (Python): {(t3-t2)*1000:.2f} ms total -> {((t3-t2)*1000)/batch_size:.2f} ms/eval")

# 3. Testear A* Puro (Emulando evaluate(ind) del Evaluator C++)
sok_path = "scratch/dummy_shell5.sok"
os.makedirs("scratch", exist_ok=True)
with open(sok_path, "w") as f:
    for _ in range(10):
        # 10 tableros iguales separados por doble salto de línea
        f.write(board_shell5.strip() + "\n\n")

print("\n[3] Ejecutando 10 llamadas reales a A* (Hungarian, max 5.0s)...")
env = os.environ.copy()
env["MAX_SECONDS"] = "5.0"
t4 = time.time()
# batch_solver corre secuencialmente la lista. 1 es para calc_branching
proc = subprocess.run(["./build/batch_solver", sok_path, "hungarian", "1"], env=env, capture_output=True, text=True)
t5 = time.time()

if proc.returncode != 0:
    print(f"Error en batch_solver:\n{proc.stderr}")
else:
    output = proc.stdout
    timeouts = output.count("TIMEOUT")
    solved = output.count("SOLVED")
    oom = output.count("OOM")
    print(f"Resultado A* 10 llamadas: {solved} SOLVED, {timeouts} TIMEOUT, {oom} OOM")
    print(f"Tiempo Total A*: {(t5-t4)*1000:.2f} ms -> {((t5-t4)*1000)/10:.2f} ms/eval")
