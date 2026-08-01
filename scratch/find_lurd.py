import sys
import pandas as pd
import subprocess
import os

sys.path.append('surrogate_models')
from prepare_path_consistency import parse_sok_files, build_fold_map

fold_map = build_fold_map()
records = parse_sok_files("training_data/Solvables", fold_map)
records = records[0::10]  # Part 0

sok_path = "temp_find_lurd.sok"
tsv_path = "temp_find_lurd.tsv"

target_lurd = "lluuuuuurrrrddrrururrdrruulDllldldlluuluuullluRurDldRRRldddrddrrururrrrdLulDDuururruruulDDDlddlldDDrdLuuuuurrRldddlddlLL"

for i, record in enumerate(records):
    board_str = record['board_str']
    with open(sok_path, "w") as f:
        f.write(f"Test\n")
        f.write(f"{board_str}\n\n")
        
    cmd = ["build/batch_solver", sok_path, "hungarian", tsv_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2)
        df = pd.read_csv(tsv_path, sep='\t')
        if len(df) == 0: continue
        row = df.iloc[0]
        if row['Status'] == 'SOLVED' and str(row['LURD_Path']) == target_lurd:
            print(f"FOUND MATCH AT INDEX {i}!")
            print(board_str)
            break
    except Exception:
        continue
