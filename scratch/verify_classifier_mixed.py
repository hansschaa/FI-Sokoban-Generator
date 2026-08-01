import requests
import json
import torch
import numpy as np

def read_pairs(filename, label, max_pairs=500):
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
            "board": '\n'.join(mutated),
            "true_label": label
        })
        if len(pairs) >= max_pairs: break
    return pairs

pairs_1 = read_pairs('training_data/ContrastivePairs/label_1_solvables.sok', label=1, max_pairs=500)
pairs_0 = read_pairs('training_data/ContrastivePairs/label_0_deadlocks.sok', label=0, max_pairs=500)

all_pairs = pairs_1 + pairs_0
print(f"Loaded {len(pairs_1)} solvable pairs and {len(pairs_0)} deadlock pairs.")

payload = {"boards": all_pairs}
resp = requests.post("http://127.0.0.1:5000/evaluate", json=payload)
results = resp.json()

if isinstance(results, list):
    correct = 0
    approved_1 = 0
    approved_0 = 0
    probs_1 = []
    probs_0 = []
    
    for i, r in enumerate(results):
        is_solvable = r["is_solvable"]
        prob = r["is_solvable_prob"] if "is_solvable_prob" in r else (1.0 if is_solvable else 0.0)
        true_label = all_pairs[i]["true_label"]
        
        if true_label == 1:
            probs_1.append(prob)
            if is_solvable: approved_1 += 1
        else:
            probs_0.append(prob)
            if is_solvable: approved_0 += 1
            
        if (is_solvable and true_label == 1) or (not is_solvable and true_label == 0):
            correct += 1

    print(f"Total Accuracy: {correct} out of {len(all_pairs)} ({correct/len(all_pairs)*100:.2f}%)")
    print(f"Solvables Approved: {approved_1} / {len(pairs_1)} ({approved_1/len(pairs_1)*100:.2f}%)")
    print(f"Deadlocks Approved: {approved_0} / {len(pairs_0)} ({approved_0/len(pairs_0)*100:.2f}%)")
    
    if len(probs_1) > 0:
        print(f"Solvables Probs - Min: {min(probs_1):.4f}, Max: {max(probs_1):.4f}, Mean: {sum(probs_1)/len(probs_1):.4f}")
    if len(probs_0) > 0:
        print(f"Deadlocks Probs - Min: {min(probs_0):.4f}, Max: {max(probs_0):.4f}, Mean: {sum(probs_0)/len(probs_0):.4f}")
else:
    print(results)
