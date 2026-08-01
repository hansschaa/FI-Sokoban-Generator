lines = [
    ("shell_1", 28), ("shell_1", 30), ("shell_1", 30), ("shell_1", 30), ("shell_1", 30),
    ("shell_1", 30), ("shell_1", 29), ("shell_1", 30), ("shell_1", 30), ("shell_1", 56),
    ("shell_2", 0), ("shell_2", 1), ("shell_2", 0), ("shell_2", 1), ("shell_2", 1),
    ("shell_2", 0), ("shell_2", 0), ("shell_2", 1), ("shell_2", 1), ("shell_2", 1),
    ("shell_3", 9), ("shell_3", 4), ("shell_3", 2), ("shell_3", 8), ("shell_3", 6),
    ("shell_3", 18), ("shell_3", 10), ("shell_3", 20), ("shell_3", 5), ("shell_3", 37),
    ("shell_4", 0), ("shell_4", 0), ("shell_4", 1), ("shell_4", 1), ("shell_4", 2),
    ("shell_4", 0), ("shell_4", 7), ("shell_4", 1), ("shell_4", 9), ("shell_4", 1),
    ("shell_5", 7), ("shell_5", 5), ("shell_5", 4), ("shell_5", 5)
]

total = sum(v for _, v in lines)
print(f"Total triggers: {total} in {len(lines)} runs")
print(f"Avg: {total/len(lines):.2f}")

shells = {}
for s, v in lines:
    shells.setdefault(s, []).append(v)
    
for s, vals in sorted(shells.items()):
    print(f"{s}: {sum(vals)/len(vals):.2f} (from {len(vals)} runs)")
