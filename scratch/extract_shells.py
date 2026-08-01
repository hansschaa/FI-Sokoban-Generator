import re

def main():
    boards = []
    current_grid = []
    
    with open('sok_files/benchmark_stratified_heldout.sok', 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.startswith('; board_id='):
                if current_grid:
                    boards.append(current_grid)
                    current_grid = []
            elif line.startswith('#'):
                current_grid.append(line)
            elif line == '' and current_grid:
                boards.append(current_grid)
                current_grid = []
                
    if current_grid:
        boards.append(current_grid)
        
    print(f"Loaded {len(boards)} boards.")
    
    # We want 5 shells. Let's pick boards 0, 8, 16, 24, 32 
    # to span different difficulties.
    target_indices = [0, 8, 16, 24, 32]
    
    for i, idx in enumerate(target_indices):
        grid = boards[idx]
        shell = []
        for row in grid:
            # Replace boxes (*, $), goals (., *, +) and player (@, +) with empty space
            row = row.replace('$', ' ').replace('*', ' ').replace('.', ' ').replace('@', ' ').replace('+', ' ')
            shell.append(row)
            
        with open(f"levels/shell_{i+1}.sok", 'w') as f:
            f.write("\n".join(shell) + "\n")
            
        print(f"Saved shell {i+1} from board {idx}")

if __name__ == '__main__':
    main()
