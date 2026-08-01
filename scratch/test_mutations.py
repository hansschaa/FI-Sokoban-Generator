import sys
import subprocess
import os
import uuid
import random

def mutate_board_randomly(board_str):
    board = [list(line) for line in board_str.strip().split('\n')]
    if not board: return board_str
    
    empty_positions = []
    wall_positions = []
    
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] == ' ':
                empty_positions.append((r, c))
            elif board[r][c] == '#':
                wall_positions.append((r, c))
                
    if not empty_positions: return board_str
    
    # 50% chance to add a wall, 50% chance to remove a wall
    if random.random() < 0.5 and empty_positions:
        r, c = random.choice(empty_positions)
        board[r][c] = '#'
    elif wall_positions:
        r, c = random.choice(wall_positions)
        board[r][c] = ' '
        
    return '\n'.join([''.join(row) for row in board])

def solve_board_with_cpp(board_str):
    try:
        fname = f"/tmp/board_{uuid.uuid4().hex}.sok"
        with open(fname, "w") as f:
            f.write(board_str)
            
        process = subprocess.Popen(
            ['./build/sokoban_solver', fname, '0', '1000'], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            os.remove(fname)
            print("Timed out!")
            return -1
        os.remove(fname)
        
        out_val = -1
        for line in stdout.split('\n'):
            if line.startswith("Pushes:"):
                out_val = int(line.split(":")[1].strip())
                break
                
        if out_val == -1:
            print(f"FAILED. STDOUT: {stdout}\nSTDERR: {stderr}")
        return out_val
    except Exception as e:
        print(f"Exception: {e}")
        return -1

board_str = """#################
# ##          ..#
# ##  $ $  #. # #
#      ### .$ . #
#  $# #     @. ##
###    # $ #   ##
### # ####  $ ###
##  # ## #   # ##
## #      ##    #
##### ## ### # ##
#################"""

print(f"Original pushes: {solve_board_with_cpp(board_str)}")
for i in range(10):
    mut = mutate_board_randomly(board_str)
    p = solve_board_with_cpp(mut)
    print(f"Mut {i} pushes: {p}")
