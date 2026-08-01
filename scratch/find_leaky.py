import sys
import json
sys.path.append('surrogate_models')
from prepare_path_consistency import simulate_path, build_fold_map, parse_sok_files

def run():
    fold_map = build_fold_map()
    records = parse_sok_files("../training_data/Solvables", fold_map)
    print(f"Loaded {len(records)} records")

if __name__ == '__main__':
    run()
