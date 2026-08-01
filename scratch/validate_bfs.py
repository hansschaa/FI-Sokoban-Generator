import os
import subprocess
import tempfile

def get_valid_children(board_str):
    lines = [list(l) for l in board_str.splitlines()]
    H = len(lines)
    if H == 0: return []
    W = max(len(l) for l in lines)
    for i in range(H):
        lines[i] += [' '] * (W - len(lines[i]))
        
    px, py = -1, -1
    for r in range(H):
        for c in range(W):
            if lines[r][c] in ['@', '+']:
                px, py = r, c
                break
        if px != -1: break
        
    reachable = set()
    q = [(px, py)]
    visited = set([(px, py)])
    while q:
        x, y = q.pop(0)
        reachable.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                # LINEA EXACTA PIDIDA POR EL USUARIO:
                if lines[nx][ny] in [' ', '.', '@', '+']:
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    
    children = []
    for r in range(H):
        for c in range(W):
            if lines[r][c] in ['$', '*']:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    px_req, py_req = r - dx, c - dy
                    nx, ny = r + dx, c + dy
                    if (px_req, py_req) in reachable:
                        if 0 <= nx < H and 0 <= ny < W and lines[nx][ny] in [' ', '.']:
                            children.append("child")
    return children

def run_cpp_test(board_str):
    # Surround with walls to prevent C++ from parsing errors
    fd, temp_filename = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, 'w') as f:
        f.write(board_str)
        
    cmd = ['./build/test_solver', temp_filename, 'simple']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        out = result.stdout
        for line in out.splitlines():
            if line.startswith('total_children:'):
                return int(line.split(':')[1].strip())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.remove(temp_filename)
    return -1

if __name__ == "__main__":
    boards = [
        # Caso 1: Push simple (1 hijo)
        "#######\n#@ $ .#\n#######",
        
        # Caso 2: Caja bloquea al jugador por la izquierda (0 hijos)
        # Jugador no puede llegar a la izquierda de la caja derecha.
        "#######\n#@ $$.#\n#######",
        
        # Caso 3: Push bloqueado por pared
        "#######\n#@    #\n# $ $ #\n### . #\n#######",
        
        # Caso 4: Caja en la meta, se puede empujar para sacarla
        "#######\n#@  * #\n#     #\n#   . #\n#######",
        
        # Caso 5: Jugador encerrado totalmente
        "#######\n# $$  #\n#$@$  #\n# $$  #\n#######"
    ]
    
    for i, board in enumerate(boards):
        print(f"--- Caso {i+1} ---")
        print(board)
        py_c = len(get_valid_children(board))
        cpp_c = run_cpp_test(board)
        print(f"Python BFS Hijos: {py_c}")
        print(f"C++ Engine Hijos: {cpp_c}")
        print("Coinciden: ", py_c == cpp_c)
        print()
