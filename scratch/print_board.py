board = """########
#    ###
# # .###
#    $ #
# #$$  #
# #.$# #
# #  @ #
# ###  #
#.#### #
#.  #  #
########"""
lines = board.split('\n')
for r, line in enumerate(lines):
    print(f"Row {r:2d}: {line}")
