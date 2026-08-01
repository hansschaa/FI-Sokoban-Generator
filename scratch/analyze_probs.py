import numpy as np
with open("scratch/probs.log", "r") as f:
    probs = [float(x.strip()) for x in f.readlines()]
probs = np.array(probs)
print(f"Total Evaluations: {len(probs)}")
if len(probs) == 0:
    print("No evaluations found.")
    exit(0)
print(f"Min: {probs.min():.6f}")
print(f"Max: {probs.max():.6f}")
print(f"Mean: {probs.mean():.6f}")
print(f"Median: {np.median(probs):.6f}")
print("\nHistogram:")
hist, bins = np.histogram(probs, bins=10, range=(0.0, 1.0))
for i in range(len(hist)):
    print(f"{bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]}")
print(f"\nProbs >= 0.65 (Approved): {np.sum(probs >= 0.65)}")
