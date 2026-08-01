import json
import requests

payload = {
    "boards": [
        {"board": "", "parent_board": ""}
    ]
}

res = requests.post("http://127.0.0.1:5000/evaluate", json=payload)
print(res.status_code)
print(res.text)
