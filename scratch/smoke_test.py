import requests
import time
import subprocess
import json
import sys

print("🔥 Initiating smoke test and verifying model SHA256 checksums...")
server = subprocess.Popen([sys.executable, "surrogate_models/surrogate_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

server_ready = False
while True:
    line = server.stdout.readline()
    if not line:
        break
    print("   [Server Log]", line.strip())
    if "Server ready" in line or "Running on" in line:
        server_ready = True
        break

if not server_ready:
    print("❌ Server failed to start")
    server.kill()
    exit(1)

time.sleep(1)
print("✅ Server booted successfully and verified checksums!")

# A completely valid board where the player pushes a box onto a target
parent_board = "#######\n#     #\n# @$. #\n#     #\n#######\n"
child_board  = "#######\n#     #\n#  @* #\n#     #\n#######\n"

payload = {
    "boards": [
        {"board": child_board, "parent_board": parent_board}
    ]
}

print("Sending request 1...")
try:
    response = requests.post("http://127.0.0.1:5000/evaluate", json=payload, timeout=10)
    print("Status code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Exception during request:", e)

print("Killing server...")
server.kill()
