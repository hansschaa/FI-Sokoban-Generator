import requests
import json
import torch
import numpy as np

def read_pairs(filename, max_pairs=1000):
    pairs = []
    with open(filename, 'r') as f:
        content = f.read().split('\n\n')
    
    for block in content:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        if len(lines) < 4: continue
        
        source = []
        mutated = []
        parsing_source = False
        parsing_mutated = False
        
        for l in lines:
            if l.startswith('source_board:'):
                parsing_source = True
                parsing_mutated = False
            elif l.startswith('mutated_board:'):
                parsing_source = False
                parsing_mutated = True
            elif l.startswith('label:'):
                parsing_source = False
                parsing_mutated = False
            else:
                if parsing_source: source.append(l)
                if parsing_mutated: mutated.append(l)
        
        pairs.append({
            "parent_board": '\n'.join(source),
            "board": '\n'.join(mutated)
        })
        if len(pairs) >= max_pairs: break
    return pairs

pairs = read_pairs('training_data/ContrastivePairs/label_1_solvables.sok', 1000)
print(f"Loaded {len(pairs)} solvable pairs.")

payload = {"boards": pairs}
resp = requests.post("http://127.0.0.1:5001/evaluate", json=payload)
results = resp.json()

if isinstance(results, list):
    approved = sum(1 for r in results if r["is_solvable"])
    print(f"Server Approved (Solvable): {approved} out of {len(pairs)} ({approved/len(pairs)*100:.2f}%)")
else:
    print(results)
