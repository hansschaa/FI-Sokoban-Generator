import glob
files = glob.glob("scratch/temp_pilot_*.csv")
print("Found", len(files), "csv files.")
disyuntor = 0
for f in files:
    with open(f) as file:
        lines = file.readlines()
        # The circuit breaker is usually reached when there are ~30 lines
        # if maxEvals is 1000000 and timeLimit is 300
        # Wait, if it didn't hit circuit breaker, it would have way more lines!
        if len(lines) <= 100: 
            disyuntor += 1
print("Disyuntor fired (approx):", disyuntor)
