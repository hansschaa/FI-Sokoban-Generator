#include "path_simulator.h"
#include <cctype>
#include <functional>

using namespace std;

PathSimulator::SimState::SimState(const string& flat_board, int rows, int cols) {
    h = rows;
    w = cols;
    grid.resize(h, string(w, ' '));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            grid[y][x] = flat_board[y * w + x];
        }
    }
}

vector<char> PathSimulator::SimState::get_legal_movements() const {
    vector<char> moves;
    int px = -1, py = -1;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (grid[y][x] == '@' || grid[y][x] == '+') {
                px = x; py = y; break;
            }
        }
        if (px != -1) break;
    }

    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};
    char move_chars[] = {'u', 'd', 'l', 'r'};
    char push_chars[] = {'U', 'D', 'L', 'R'}; // Letras mayúsculas para empujes

    for (int i = 0; i < 4; i++) {
        int nx = px + dx[i];
        int ny = py + dy[i];

        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;

        char c = grid[ny][nx];
        
        // 1. Caminar: Si la adyacente es espacio vacío o meta
        if (c == ' ' || c == '.') {
            moves.push_back(move_chars[i]);
        }
        // 2. Empujar: Si la adyacente es una caja
        else if (c == '$' || c == '*') {
            int nnx = nx + dx[i];
            int nny = ny + dy[i];
            
            // Verificar que no empujemos fuera del mapa
            if (nnx >= 0 && nny >= 0 && nnx < w && nny < h) {
                char beyond = grid[nny][nnx];
                // Si la celda detrás de la caja está libre, es un empuje legal
                if (beyond == ' ' || beyond == '.') {
                    moves.push_back(push_chars[i]);
                }
            }
        }
    }
    return moves;
}

unique_ptr<PathSimulator::SimState> PathSimulator::SimState::apply_move(char m) const {
    auto next_state = make_unique<SimState>(*this);
    int px = -1, py = -1;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (next_state->grid[y][x] == '@' || next_state->grid[y][x] == '+') {
                px = x; py = y; break;
            }
        }
        if (px != -1) break;
    }

    int dx = 0, dy = 0;
    bool is_push = (m >= 'A' && m <= 'Z');
    char m_lower = tolower(m);

    if (m_lower == 'u') { dx = 0; dy = -1; }
    else if (m_lower == 'd') { dx = 0; dy = 1; }
    else if (m_lower == 'l') { dx = -1; dy = 0; }
    else if (m_lower == 'r') { dx = 1; dy = 0; }
    else return nullptr; 

    int nx = px + dx;
    int ny = py + dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= h) return nullptr;

    char target = next_state->grid[ny][nx];

    if (is_push) {
        if (target != '$' && target != '*') return nullptr;
        int nnx = nx + dx, nny = ny + dy;
        if (nnx < 0 || nny < 0 || nnx >= w || nny >= h) return nullptr;
        char beyond = next_state->grid[nny][nnx];
        if (beyond != ' ' && beyond != '.') return nullptr;

        next_state->grid[ny][nx] = (target == '*') ? '.' : ' ';
        next_state->grid[nny][nnx] = (beyond == '.') ? '*' : '$';
    } else {
        if (target != ' ' && target != '.') return nullptr;
    }

    char p_char = next_state->grid[py][px];
    next_state->grid[py][px] = (p_char == '+') ? '.' : ' ';
    char new_target = next_state->grid[ny][nx]; 
    next_state->grid[ny][nx] = (new_target == '.') ? '+' : '@';

    return next_state;
}

string PathSimulator::SimState::to_string_key() const {
    string key;
    for (const auto& row : grid) key += row + '\n';
    return key;
}

bool PathSimulator::SimState::is_pattern2_deadlock() const {
    auto is_wall = [&](int yy, int xx) {
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) return true;
        return grid[yy][xx] == '#';
    };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            char c = grid[y][x];

            // --- NUEVO: DETECCIÓN DE DEADLOCK 2x2 ---
            // Revisa si hay 4 cajas formando un cuadrado (2x2)
            if ((c == '$' || c == '*') && y + 1 < h && x + 1 < w) {
                char c_right = grid[y][x + 1];
                char c_down  = grid[y + 1][x];
                char c_diag  = grid[y + 1][x + 1];

                // Si las 3 casillas vecinas también tienen cajas
                if ((c_right == '$' || c_right == '*') && 
                    (c_down  == '$' || c_down  == '*') && 
                    (c_diag  == '$' || c_diag  == '*')) {
                    
                    // Si al menos UNA caja no está en una meta ('$'), el bloque 2x2 es un deadlock
                    if (c == '$' || c_right == '$' || c_down == '$' || c_diag == '$') {
                        return true;
                    }
                }
            }
            // ----------------------------------------

            // Solo hacemos las detecciones de paredes si la celda actual es una caja normal
            if (c != '$') continue;

            // 1. Deadlock de Esquina (Corner)
            bool up = is_wall(y - 1, x);
            bool down = is_wall(y + 1, x);
            bool left = is_wall(y, x - 1);
            bool right = is_wall(y, x + 1);

            if ((up && left) || (up && right) || (down && left) || (down && right)) {
                return true;
            }

            // 2. Deadlock de Borde con 2 Cajas (Edge 2-Box)
            if ((left || right) && ((y + 1 < h && grid[y + 1][x] == '$') || (y - 1 >= 0 && grid[y - 1][x] == '$'))) {
                return true;
            }
            if ((up || down) && ((x + 1 < w && grid[y][x + 1] == '$') || (x - 1 >= 0 && grid[y][x - 1] == '$'))) {
                return true;
            }
        }
    }
    return false;
}

PathBranchingStats PathSimulator::compute_stats(const string& flat_initial_board, int rows, int cols, const string& lurd) {
    PathBranchingStats stats;
    SimState initial(flat_initial_board, rows, cols);

    // --- Compute Real Branching ---
    auto cur_real = make_unique<SimState>(initial);
    for (char step : lurd) {
        vector<char> moves = cur_real->get_legal_movements();
        int children = moves.size();

        stats.branching_real_total_nodes += children;
        stats.branching_real_min = min(stats.branching_real_min, children);
        stats.branching_real_max = max(stats.branching_real_max, children);
        stats.states++;

        cur_real = cur_real->apply_move(step);
        if (!cur_real) break;
    }
    if (stats.branching_real_min == 2147483647) {
        stats.branching_real_min = 0;
        stats.branching_real_max = 0;
    }

    // --- Compute Effective Branching ---
    auto cur_eff = make_unique<SimState>(initial);
    unordered_set<string> seen_children;

    for (char step : lurd) {
        if (!isalpha(step)) continue;

        vector<char> moves = cur_eff->get_legal_movements();
        int effective_this_state = 0;

        for (char m : moves) {
            stats.total_children_generated++;

            auto child = cur_eff->apply_move(m);
            if (!child) continue;

            string key = child->to_string_key();

            if (seen_children.count(key)) {
                stats.repeated_nodes++;
                continue;
            }

            seen_children.insert(key);

            if (child->is_pattern2_deadlock()) {
                stats.deadlocks++;
                continue;
            }

            effective_this_state++;
        }

        stats.branching_effective_total_nodes += effective_this_state;
        stats.branching_effective_min = min(stats.branching_effective_min, effective_this_state);
        stats.branching_effective_max = max(stats.branching_effective_max, effective_this_state);

        cur_eff = cur_eff->apply_move(step);
        if (!cur_eff) break;
    }

    if (stats.branching_effective_min == 2147483647) {
        stats.branching_effective_min = 0;
        stats.branching_effective_max = 0;
    }

    return stats;
}