import os
import glob
import subprocess

SOK_DIR = "training_data/Solvables"
files = glob.glob(os.path.join(SOK_DIR, "**/*.sok"), recursive=True)[:5]

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

for i, f in enumerate(files):
    with open(f, 'r') as file:
        board_str = file.read()
    
    py_c = len(get_valid_children(board_str))
    
    cmd = ['./build/test_solver', f, 'simple']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        cpp_c = -1
        for line in result.stdout.splitlines():
            if line.startswith('INITIAL_CHILDREN:'):
                cpp_c = int(line.split(':')[1].strip())
    except Exception as e:
        cpp_c = -1
        
    print(f"--- Tablero Real {i+1} ---")
    print(board_str.strip())
    print(f"Python BFS Hijos: {py_c}")
    print(f"C++ Engine Hijos: {cpp_c}")
    print("Coinciden: ", py_c == cpp_c)
    print()

