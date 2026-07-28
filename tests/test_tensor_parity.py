import sys
import numpy as np

sys.path.append('surrogate_models/data')
from board_utils import encode_board

def extract_boards(sok_file, n=1):
    with open(sok_file, 'r') as f:
        content = f.read()
    
    boards = []
    current_board = []
    for line in content.split('\n'):
        if line.strip().startswith('#'):
            current_board.append(line)
        elif line.strip() == '' and current_board:
            boards.append('\n'.join(current_board))
            current_board = []
            if len(boards) >= n:
                break
    if current_board and len(boards) < n:
        boards.append('\n'.join(current_board))
    return boards

def main():
    boards = extract_boards('tests/test_board.txt', 1)
    
    for i, board_str in enumerate(boards):
        tensor = encode_board(board_str, max_h=25, max_w=25)
        # Tensor is (6, 25, 25)
        
        print(f"--- BOARD {i} TENSORS ---")
        for ch in range(6):
            print(f"Channel {ch}:")
            for r in range(25):
                row_str = ''
                for c in range(25):
                    row_str += '1' if tensor[ch, r, c] > 0.5 else '0'
                print(row_str)

if __name__ == '__main__':
    main()
