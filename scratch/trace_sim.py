import sys
import time

board_str = """#################
##  .  # ##### ##
### .##   #@    #
##   ### ### # ##
#  #            #
#  #  . # #   # #
#  #   #####    #
#   #     ###$# #
# #### ##  #  #.#
# ###  ## $   # #
#        ## # ###
# $$##  ##    ###
#   ##   ## #   #
#################"""

lurd = "rddllllldlddrrrddRulullldddlllllddrrUdlluuuuuuuururrdddrdrrrrddrRlulullluuullullddddldddrRRllluuuuuururrdddrdrrrrddrrddrruuuUddddlluuRllulullluuullullddddldddrrrRuruurrrddrrddrruUlllululllddDrddlUUllllluuuuuururrdddrdrrrrddrrrrUdlllulullluuurrrrrdrdRuulllllldlddrrrddrrrurUddlllulullluuullullddddldddddrUrurrrUUddllldlluRRRRuruUUdrrrddrrruruUUlullllldLurrrrrruurrddLdllulllldlLurrrrrrdrruLddrdddlllululllullUdrdrrrrddrrruruuuulLLrdrdrdddlllulullluullUUdddrdrrrrddrrruruuuulllLLrrrdrdrdddlllulullluuulluUddddrdrrrrddrrruruuuulllllLLLdlUddrdrddDuuuuuurrrrrdrdrRluurrdDDuuulllllllldldddddrddlUUUUUUluulU"

def simulate_path_trace(board_str, lurd_path):
    lines = [list(line) for line in board_str.splitlines()]
    px, py = -1, -1
    for r, row in enumerate(lines):
        for c, char in enumerate(row):
            if char in ['@', '+']:
                px, py = r, c
                break
        if px != -1: break
        
    dirs = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1),
            'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
            
    for i, m in enumerate(lurd_path):
        dx, dy = dirs[m]
        nx, ny = px + dx, py + dy
        is_push = m.isupper()
        
        # Check before moving
        if nx < 0 or ny < 0 or nx >= len(lines) or ny >= len(lines[nx]):
            print(f"OUT OF BOUNDS at step {i} (move {m})!")
            print(f"Player at ({px}, {py}), trying to move to ({nx}, {ny})")
            sys.exit(1)
            
        target_char = lines[nx][ny]
        if target_char == '#':
            print(f"CRASH INTO WALL at step {i} (move {m})!")
            print(f"Player at ({px}, {py}), trying to move to ({nx}, {ny})")
            sys.exit(1)
            
        if is_push:
            bx, by = nx + dx, ny + dy
            if bx < 0 or by < 0 or bx >= len(lines) or by >= len(lines[bx]):
                print(f"BOX OUT OF BOUNDS at step {i} (move {m})!")
                sys.exit(1)
            if lines[bx][by] in ['#', '$', '*']:
                print(f"BOX CRASH at step {i} (move {m})!")
                sys.exit(1)
                
            box_char = lines[nx][ny]
            lines[nx][ny] = '-' if box_char == '$' else '.'
            lines[bx][by] = '$' if lines[bx][by] in [' ', '-'] else '*'
            
        p_char = lines[px][py]
        lines[px][py] = ' ' if p_char == '@' else '.'
        n_char = lines[nx][ny]
        lines[nx][ny] = '@' if n_char in [' ', '-'] else '+'
        px, py = nx, ny
        
    print("SUCCESS")

simulate_path_trace(board_str, lurd)
