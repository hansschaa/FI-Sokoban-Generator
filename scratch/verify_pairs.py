import torch
import numpy as np

pairs = torch.load('surrogate_models/results/siamese_ranknet_test_heldout.pt', weights_only=False)
print("=== VERIFICACIÓN MANUAL DE 5 PARES ===")
for i in range(5):
    p = pairs[i]
    rA = p['raw_A']
    rB = p['raw_B']
    nA = p['norm_A']
    nB = p['norm_B']
    
    # Lógica exacta del entrenamiento
    target = 1.0 if nA > nB else -1.0
    
    # Lógica conceptual
    real_sign = 1.0 if rA > rB else -1.0
    
    print(f"Par {i+1}:")
    print(f"  Pushes reales: A={rA}, B={rB}")
    print(f"  Norms: A={nA:.4f}, B={nB:.4f}")
    print(f"  Target que recibe MarginRankingLoss: {target}")
    print(f"  Signo correcto calculado a mano  : {real_sign}")
    print(f"  ¿Coinciden? {'SÍ' if target == real_sign else 'NO'}")
