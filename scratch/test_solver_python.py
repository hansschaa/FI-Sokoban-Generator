import subprocess

def solve_board_with_cpp(board_str):
    try:
        process = subprocess.Popen(
            ['./build/sokoban_solver', '0'], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=board_str)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        for line in stdout.split('\n'):
            if line.startswith("Pushes:"):
                return int(line.split(":")[1].strip())
        return -1
    except Exception as e:
        return -1

b = """#################
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

print(solve_board_with_cpp(b))
