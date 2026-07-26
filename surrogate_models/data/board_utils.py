import numpy as np

MAX_H = 25
MAX_W = 25

def compute_deadlock_mask(lines, max_h=MAX_H, max_w=MAX_W):
    H = len(lines)
    W = max(len(l) for l in lines) if lines else 0
    H = min(H, max_h)
    W = min(W, max_w)

    board = np.full((H, W), ' ', dtype=str)
    for r, line in enumerate(lines[:H]):
        for c, ch in enumerate(line[:W]):
            board[r, c] = ch

    mask = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            ch = board[i, j]
            # Excluir muros y metas de la máscara de deadlock (una meta nunca es deadlock)
            if ch in [' ', '$', '@']:
                top_wall = (i == 0 or board[i-1, j] == '#')
                bottom_wall = (i == H-1 or board[i+1, j] == '#')
                left_wall = (j == 0 or board[i, j-1] == '#')
                right_wall = (j == W-1 or board[i, j+1] == '#')
                
                vert_blocked = top_wall or bottom_wall
                horiz_blocked = left_wall or right_wall
                
                if vert_blocked and horiz_blocked:
                    mask[i, j] = 1.0 # Corner
                elif vert_blocked:
                    trapped = True
                    # Scan left
                    for x in range(j - 1, -1, -1):
                        if board[i, x] == '#': break
                        if board[i, x] in ['.', '*', '+']: trapped = False; break
                        if not ((i > 0 and board[i-1, x] == '#') or (i < H-1 and board[i+1, x] == '#')): trapped = False; break
                    # Scan right
                    for x in range(j + 1, W):
                        if board[i, x] == '#': break
                        if board[i, x] in ['.', '*', '+']: trapped = False; break
                        if not ((i > 0 and board[i-1, x] == '#') or (i < H-1 and board[i+1, x] == '#')): trapped = False; break
                    if trapped: mask[i, j] = 1.0
                elif horiz_blocked:
                    trapped = True
                    # Scan up
                    for y in range(i - 1, -1, -1):
                        if board[y, j] == '#': break
                        if board[y, j] in ['.', '*', '+']: trapped = False; break
                        if not ((j > 0 and board[y, j-1] == '#') or (j < W-1 and board[y, j+1] == '#')): trapped = False; break
                    # Scan down
                    for y in range(i + 1, H):
                        if board[y, j] == '#': break
                        if board[y, j] in ['.', '*', '+']: trapped = False; break
                        if not ((j > 0 and board[y, j-1] == '#') or (j < W-1 and board[y, j+1] == '#')): trapped = False; break
                    if trapped: mask[i, j] = 1.0

    return mask

def encode_board(board_str, max_h=MAX_H, max_w=MAX_W):
    """
    Convierte un string de tablero a un tensor float32 de forma (6, max_h, max_w).
    Canales:
      0 → Muros (#)
      1 → Interior caminable (flood-fill simplificado)
      2 → Cajas ($ o *)
      3 → Metas (. o * o +)
      4 → Jugador (@ o +)
      5 → Máscara de Deadlocks Estáticos
    """
    lines = board_str.splitlines()
    H = len(lines)
    W = max(len(l) for l in lines) if lines else 0

    H = min(H, max_h)
    W = min(W, max_w)

    char_matrix = np.full((H, W), ' ', dtype=str)
    for r, line in enumerate(lines[:H]):
        for c, ch in enumerate(line[:W]):
            char_matrix[r, c] = ch

    # Tensor de 6 canales
    tensor = np.zeros((6, max_h, max_w), dtype=np.float32)
    
    # Rellenar Canal 0 con muros en el padding (exterior)
    tensor[0, :, :] = 1.0

    for r in range(H):
        for c in range(W):
            ch = char_matrix[r, c]
            # Limpiar el muro por defecto en el área jugable
            tensor[0, r, c] = 0.0

            if ch == '#':
                tensor[0, r, c] = 1.0
            elif ch == ' ':
                tensor[1, r, c] = 1.0
            elif ch == '$':
                tensor[1, r, c] = 1.0
                tensor[2, r, c] = 1.0
            elif ch == '.':
                tensor[1, r, c] = 1.0
                tensor[3, r, c] = 1.0
            elif ch == '*':
                tensor[1, r, c] = 1.0
                tensor[2, r, c] = 1.0
                tensor[3, r, c] = 1.0
            elif ch == '@':
                tensor[1, r, c] = 1.0
                tensor[4, r, c] = 1.0
            elif ch == '+':
                tensor[1, r, c] = 1.0
                tensor[3, r, c] = 1.0
                tensor[4, r, c] = 1.0
            else:
                tensor[1, r, c] = 1.0

    # Computar canal 5 (Deadlocks)
    deadlock_mask = compute_deadlock_mask(lines, max_h, max_w)
    tensor[5, :H, :W] = deadlock_mask

    return tensor

def augment_tensor(tensor):
    variants = [tensor]
    variants.append(np.flip(tensor, axis=1)) # Vertical
    variants.append(np.flip(tensor, axis=2)) # Horizontal
    variants.append(np.flip(np.flip(tensor, axis=1), axis=2)) # Rot 180
    
    t_T = np.transpose(tensor, axes=(0, 2, 1))
    variants.append(t_T)
    variants.append(np.flip(t_T, axis=1))
    variants.append(np.flip(t_T, axis=2))
    variants.append(np.flip(np.flip(t_T, axis=1), axis=2))
    return variants
