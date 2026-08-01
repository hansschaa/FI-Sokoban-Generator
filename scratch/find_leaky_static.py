import sys
import json

sys.path.append('surrogate_models')
from prepare_path_consistency import parse_sok_files, build_fold_map

fold_map = build_fold_map()
records = parse_sok_files("training_data/Solvables", fold_map)
print(f"Loaded {len(records)} records")

leaky_boards = []

for i, record in enumerate(records):
    board_str = record['board_str']
    lines = board_str.splitlines()
    
    # Check if there is any walkable space that is on the very edge of the strings,
    # OR if we do a flood fill from outside, does it reach the interior?
    
    # Let's pad it with a known exterior character 'E'
    max_w = max(len(l) for l in lines) if lines else 0
    grid = []
    for l in lines:
        row = list(l) + ['E'] * (max_w - len(l))
        grid.append(row)
        
    H = len(grid)
    W = max_w
    
    is_leaky = False
    
    # Any floor tile on the absolute edges is a leak
    for r in range(H):
        for c in range(W):
            if grid[r][c] in [' ', '.', '@', '+', '$', '*']:
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    is_leaky = True
                    break
                # Also if it touches 'E', it's leaky!
                if (r > 0 and grid[r-1][c] == 'E') or \
                   (r < H-1 and grid[r+1][c] == 'E') or \
                   (c > 0 and grid[r][c-1] == 'E') or \
                   (c < W-1 and grid[r][c+1] == 'E'):
                    is_leaky = True
                    break
        if is_leaky: break
        
    if is_leaky:
        leaky_boards.append((i, board_str))

print(f"\nEncontrados {len(leaky_boards)} tableros corruptos (con fugas).")
if leaky_boards:
    print(f"\nAQUÍ ESTÁ EL PRIMER TABLERO CORRUPTO (Index {leaky_boards[0][0]}):")
    print("="*40)
    print(leaky_boards[0][1])
    print("="*40)
