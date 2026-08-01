import sys
sys.path.append('surrogate_models')
from prepare_path_consistency import simulate_path

# Un tablero asimétrico donde las líneas de abajo son más cortas
board_str = """#######
#     #
# $ . #
#   ###
#   
###"""

print("--- TABLERO ORIGINAL (Nótese las líneas cortas) ---")
for i, line in enumerate(board_str.splitlines()):
    print(f"Línea {i}: '{line}' (len={len(line)})")

print("\n--- SIMULANDO (El padding ocurrirá internamente) ---")
# lurd falso solo para que no crashee
states = simulate_path(board_str, "R")

print("\n--- RESULTADO (Estado 0 extraído) ---")
padded_str = states[0][0]
for i, line in enumerate(padded_str.splitlines()):
    print(f"Línea {i}: '{line}' (len={len(line)})")
