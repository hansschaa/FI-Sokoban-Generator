import sys
import pandas as pd
import glob
import os

TSV_FILE = "scratch/path_consistency_results.tsv"
SOK_DIR = "training_data/Solvables"

df = pd.read_csv(TSV_FILE, sep='\t')
print("DF length:", len(df))

files = glob.glob(os.path.join(SOK_DIR, "**/*.sok"), recursive=True)
print("Files found:", len(files))

board_map = {}
for f in files[:100]:
    with open(f, 'r') as file:
        name = os.path.basename(f).replace('.sok', '')
        board_map[name] = True
print("Mapped:", len(board_map))

matches = 0
for idx, row in df.iterrows():
    if row['Status'] != 'SOLVED': continue
    if row['LURD_Path'] == 'NONE': continue
    name = str(row['LevelName']).split(' - ')[0].strip()
    if name in board_map:
        matches += 1
print("Matches:", matches)
