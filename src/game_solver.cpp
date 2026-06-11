#include <deque>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <queue>      // Añadido para el BFS de reconstruct_lurd
#include <string>     // Añadido para std::string
#include <iostream>
#include "game_solver.h"
#include "constant.h"
#include "locked.h"
#include "repeat.h"
#include "solver_template.h"
#include "mazesolver.h"
#include "hungarian.h"

using namespace constant;
using namespace std;

detect_legal::detect_legal(const game_node* node) {
    node->get_matrix0(matrix_with_box);
    maze_solver<Method::bfs, bool> dmaze;
    dmaze.solve(matrix_with_box, node->person_point, point(0, 0));
    value = std::move(dmaze.zero_matrix);
    (value)[node->person_point.x][node->person_point.y] = true;
}

detect_legal::detect_legal(vector<vector<char>>& matrix, point& start) {
    matrix_with_box = matrix;
    maze_solver<Method::bfs, bool> dmaze;
    dmaze.solve(matrix_with_box, start, point(0, 0));
    value = std::move(dmaze.zero_matrix);
    (value)[start.x][start.y] = true;
}

detect_legal::detect_legal() {}

bool detect_legal::can_get(point& des) {
    return (value)[des.x][des.y] == true;
}

bool detect_legal::can_box_move(
    point& box,
    point& person)
{
    point new_point;

    new_point = box * 2 - person;

    if(new_point.x < 0 ||
       new_point.x >= m ||
       new_point.y < 0 ||
       new_point.y >= n)
    {
        return false;
    }

    return matrix_with_box[new_point.x]
                          [new_point.y]
           == BLANK;
}

game_solver::game_solver(string& game_map, unsigned int mm, unsigned int nn, int memval) {
    original_map_1d = game_map; // AÑADIDO: Guardar string 1D para el simulador
    m = mm;
    n = nn;
    matrix0  = vector<vector<bool>>(mm, vector<bool>(nn, false));
    end_vec = vector<vector<bool>>(mm, vector<bool>(nn, false));
    blank_matrix = vector<vector<char>>(mm, vector<char>(nn, 0));

    point person_start;
    set<point> box_point_start;

    char temp_c;
    point temp_p;

    // FIX: int8_t causaba overflow silencioso en mapas > 127 filas/columnas.
    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            temp_c = game_map[x*n + y];
            temp_p = {x, y};
            switch (temp_c) {
            case '#':
                blank_matrix[x][y] = WALL;
                break;
            case ' ':
                blank_matrix[x][y] = BLANK;
                break;
            case '$':
                blank_matrix[x][y] = BLANK;
                box_point_start.insert(temp_p);
                break;
            case '*':
                blank_matrix[x][y] = BLANK;
                box_point_start.insert(temp_p);
                end_vec[x][y] = true;
                break;
            case '.':
                blank_matrix[x][y] = BLANK;
                end_vec[x][y] = true;
                break;
            case '@':
                blank_matrix[x][y] = BLANK;
                person_start = temp_p;
                break;
            case '+':
                blank_matrix[x][y] = BLANK;
                person_start = temp_p;
                end_vec[x][y] = true;
                break;
            default:
                break;
            }
        }
    }
    //
    // INIT MEMORY POOLS
    // my_memory_pool::init() now safely frees any previous allocation,
    // so constructing multiple game_solver instances is leak-free.
    //
    game_mem.init(sizeof(game_node), memval * 1024 * 1024 / sizeof(game_node));
    constant::maze_mp.init(sizeof(point), mm * nn * 4);
    init = game_node(box_point_start,person_start);
    lk.init();
    penalty_solver = Penalty(mm, nn);
    set_lambda_function();

}

void game_solver::vars_init(game_node& input){
    rpt.init(input);
    lk.init();
    game_mem.clear();
    stat_expanded_nodes     = 0;
    stat_total_children     = 0;
    stat_effective_children = 0;
    stat_repeated_nodes     = 0;
    stat_deadlocks          = 0;
}

void game_solver::vars_clear(game_node& input){
    rpt.zobrist_hash.erase(&input);
    for (auto tp: rpt.zobrist_hash){
        tp->~game_node();
        game_mem.deallocate((void *)tp);
    }
    rpt.zobrist_hash.clear();
    game_mem.clear();

    // --- ARREGLO DE LEAK: Vaciar y liberar capacidad de vectores anidados ---
    goal_positions.clear();
    goal_positions.shrink_to_fit();
    
    dist_to_goal.clear();
    dist_to_goal.shrink_to_fit(); // Esto destruye todas las matrices 3D del Heap inmediatamente
}

// FIX: antes recibía game_node por valor, copiando el set<point> completo
// en cada celda del doble loop m×n de Astar_init(). Ahora es const ref.
int game_solver::get_nums2(const game_node& input) {
    auto p = *(input.box_list.begin());
    if (end_vec[p.x][p.y] == true) { return 0; }

    //
    // USE A SEPARATE LOCAL MEMORY POOL
    // get_nums2 is called from Astar_init(), which runs before test_template,
    // but sharing game_mem would cause vars_init/vars_clear to clobber the
    // pool mid-solve if ever called in a nested context.
    //
    my_memory_pool local_mem;
    local_mem.init(sizeof(game_node), m * n * 4);

    repeat local_rpt;
    locked local_lk;
    // FIX: repeat::init toma game_node& (no-const); como input es const ref
    // hacemos una copia local solo para la inicialización del hash de Zobrist.
    game_node input_copy = input;
    local_rpt.init(input_copy);
    local_lk.init();

    auto local_get_neighbors = [&](const game_node* n_min, std::function<void(const game_node*)> callback) {
        detect_legal test(n_min);
        for (auto item = n_min->box_list.begin(); item != n_min->box_list.end(); item++) {
            auto box = *item;
            for (auto direction : four_direction) {
                auto new_point = box + direction;
                if (test.can_get(new_point)) {
                    if (test.can_box_move(box, new_point)) {
                        auto new_box_point = *item * 2 - new_point;
                        game_node* temp_box2 = new(local_mem.allocate()) game_node;
                        n_min->get_moved(*item, new_box_point, temp_box2);
                        vector<vector<char>> temp_matrix2;
                        temp_box2->get_matrix0(temp_matrix2);
                        if (local_lk.is_locked(new_box_point, temp_matrix2) || local_rpt.is_repeat2(temp_box2)) {
                            temp_box2->~game_node();
                            local_mem.deallocate(temp_box2);
                        } else {
                            callback(temp_box2);
                        }
                    }
                }
            }
        }
    };

    auto local_is_visited = [&](const game_node*) -> bool { return false; };
    auto local_mark_visited = [&](const game_node* n) { local_rpt.insert(n); };
    auto local_is_equal = [](const game_node* a, const game_node*) -> bool { return a->game_over(); };

    Solver_template<vector<game_node>, game_node, Method::bfs> gsolver1;
    auto resx = gsolver1.solve(&input, nullptr, local_get_neighbors, local_is_visited, local_mark_visited, local_is_equal);

    //
    // CLEANUP LOCAL POOL
    //
    local_rpt.zobrist_hash.erase(&input);
    for (auto tp : local_rpt.zobrist_hash) {
        tp->~game_node();
        local_mem.deallocate((void*)tp);
    }
    local_rpt.zobrist_hash.clear();

    if (resx.size() == 0)
    {
        return 1000;
    }
    return resx.size();
}

vector<point> game_solver::get_legal_point(vector<vector<char>>& vec, point p) {
    vector<point> result;
    vector<detect_legal> detect;
    bool flag = false;
    for (auto i = 0; i < 4; i++) {
        auto pp = p + four_direction[i];
        if (vec[pp.x][pp.y] == BLANK) {
            for (auto& dect : detect) {
                if (dect.can_get(pp)) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                flag = false;
                continue;
            }
            detect.push_back(detect_legal(vec, pp));
            result.push_back(pp);

        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// reverse_push_bfs: BFS inverso desde un único objetivo.
// ---------------------------------------------------------------------------
static void reverse_push_bfs(
    const point& goal,
    const vector<vector<char>>& blank_matrix,
    int m, int n,
    vector<vector<int>>& dist   // dist[m][n], debe estar inicializado a 1000
) {
    const int INF = 1000;

    // dist_state[x][y][d] = coste mínimo al objetivo partiendo de
    // (caja en (x,y), llegó empujada en dirección four_direction[d]).
    vector<vector<array<int,4>>> dist_state(
        m, vector<array<int,4>>(n, {INF, INF, INF, INF}));

    struct State { point box; int dir_idx; };
    deque<State> q;

    // Semillas: caja ya en el objetivo, puede haber llegado desde cualquier dir.
    for (int d = 0; d < 4; d++) {
        // El jugador debía estar en (goal - four_direction[d]) para empujar
        // la caja en la dirección d hasta el objetivo.
        point player_needed = goal - four_direction[d];
        if (player_needed.x < 0 || player_needed.x >= m ||
            player_needed.y < 0 || player_needed.y >= n) continue;
        if (blank_matrix[player_needed.x][player_needed.y] == WALL) continue;

        if (dist_state[goal.x][goal.y][d] == INF) {
            dist_state[goal.x][goal.y][d] = 0;
            q.push_back({goal, d});
        }
    }

    while (!q.empty()) {
        auto [box, dir_idx] = q.front();
        q.pop_front();

        int cur_cost = dist_state[box.x][box.y][dir_idx];

        // Para cada posible dirección de push que trajo la caja hasta box_prev:
        for (int push_d = 0; push_d < 4; push_d++) {
            point box_prev = box - four_direction[push_d];
            if (box_prev.x < 0 || box_prev.x >= m ||
                box_prev.y < 0 || box_prev.y >= n) continue;
            if (blank_matrix[box_prev.x][box_prev.y] == WALL) continue;

            // El jugador necesitaba estar al lado opuesto del push.
            point player_prev = box_prev - four_direction[push_d];
            if (player_prev.x < 0 || player_prev.x >= m ||
                player_prev.y < 0 || player_prev.y >= n) continue;
            if (blank_matrix[player_prev.x][player_prev.y] == WALL) continue;

            int new_cost = cur_cost + 1;
            if (new_cost < dist_state[box_prev.x][box_prev.y][push_d]) {
                dist_state[box_prev.x][box_prev.y][push_d] = new_cost;
                q.push_back({box_prev, push_d});
            }
        }
    }

    // Condensar: dist[x][y] = mínimo sobre las 4 direcciones de llegada.
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int best = INF;
            for (int d = 0; d < 4; d++)
                best = min(best, dist_state[i][j][d]);
            dist[i][j] = best;
        }
}

// ---------------------------------------------------------------------------
// Astar_init — versión optimizada con BFS inverso.
// ---------------------------------------------------------------------------
vector<vector<int>> game_solver::Astar_init() {
    vector<vector<int>> result(m, vector<int>(n, 0));

    goal_positions.clear();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (end_vec[i][j])
                goal_positions.push_back(point(i, j));

    int num_goals = (int)goal_positions.size();

    dist_to_goal.assign(num_goals,
        vector<vector<int>>(m, vector<int>(n, 1000)));

    for (int g = 0; g < num_goals; g++) {
        reverse_push_bfs(
            goal_positions[g],
            blank_matrix,
            m, n,
            dist_to_goal[g]
        );
    }

    penalty_solver.init(goal_positions, dist_to_goal, lk.get_side_point());

    // Tabla simple: mínimo entre objetivos (para heurística simple).
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (blank_matrix[i][j] == WALL) { result[i][j] = 1000; continue; }
            int best = 1000;
            for (int g = 0; g < num_goals; g++)
                best = min(best, dist_to_goal[g][i][j]);
            result[i][j] = best;
        }
    }

    return result;
}

void game_solver::set_lambda_function(){

    is_visited = [&](const game_node*) -> bool {
        return false;
    };

    mark_visited = [&](const game_node* n) {
        rpt.insert(n);
    };

    get_neighbors = [&](const game_node* n_min, std::function<void(const game_node*)> callback) {
        stat_expanded_nodes++;          // un nodo expandido = Java: localExpandedNodes++

        detect_legal test(n_min);
        for (auto item = n_min->box_list.begin(); item != n_min->box_list.end(); item++) {
            auto box = *item;
            for (auto direction : four_direction) {
                auto new_point = box + direction;

                stat_total_children++;  // intento de push = Java: totalChildren++

                if (test.can_get(new_point)) {
                    if (test.can_box_move(box, new_point)) {
                        auto new_box_point = *item * 2 - new_point;
                        game_node* temp_box2 = new(game_mem.allocate()) game_node;
                        n_min->get_moved(*item, new_box_point, temp_box2);
                        vector<vector<char>> temp_matrix2;
                        temp_box2->get_matrix0(temp_matrix2);

                        // --- INTEGRACIÓN DE DETECCIÓN DE DEADLOCKS ---
                        bool is_deadlocked = false;

                        if (enable_advanced_deadlocks) {
                            // 1. Chequeo básico (O(1))
                            bool basic_lock = lk.is_locked(new_box_point, temp_matrix2);
                            bool freeze_lock = false;
                            bool bipartite_lock = false;

                            // 2. Chequeo Freeze Geométrico (Recursivo, rápido)
                            if (!basic_lock) {
                                freeze_lock = lk.is_freeze_deadlock(new_box_point, temp_matrix2);
                            }

                            // 3. Chequeo Bipartito Húngaro (Costoso, solo se ejecuta si los otros fallan)
                            if (!basic_lock && !freeze_lock) {
                                bipartite_lock = is_bipartite_deadlock(*temp_box2);
                            }

                            is_deadlocked = basic_lock || freeze_lock || bipartite_lock;
                        } else {
                            // Modo FO4: Solo poda los básicos evidentes para poder contar las trampas complejas
                            is_deadlocked = lk.is_locked(new_box_point, temp_matrix2);
                        }

                        if (is_deadlocked) {
                            stat_deadlocks++;           // Java: deadlocksCount++
                            temp_box2->~game_node();
                            game_mem.deallocate(temp_box2);
                        } else if (rpt.is_repeat2(temp_box2)) {
                            stat_repeated_nodes++;      // Java: repeatedChildren++
                            temp_box2->~game_node();
                            game_mem.deallocate(temp_box2);
                        } else {
                            stat_effective_children++;  // Java: totalEffectiveChildren++
                            callback(temp_box2);
                        }
                    }
                }
            }
        }
    };

    is_equal = [](const game_node* a, const game_node*) -> bool {
        return a->game_over();
    };
}

// Función auxiliar para reconstruir LURD antes de test_template
static std::string reconstruct_lurd(const std::vector<game_node>& solution) {
    if (solution.size() <= 1) return "";

    std::string full_path = "";

    // Direcciones estándar y sus caracteres (Arriba, Abajo, Izquierda, Derecha)
    point dirs[4] = {point(-1, 0), point(1, 0), point(0, -1), point(0, 1)};
    char move_chars[4] = {'u', 'd', 'l', 'r'};
    char push_chars[4] = {'U', 'D', 'L', 'R'};

    // FIX: Iteramos en reversa porque el solver entrega la ruta [Meta ... Inicio]
    for (int i = (int)solution.size() - 1; i >= 1; i--) {
        const game_node& a = solution[i];     // Estado antes de empujar
        const game_node& b = solution[i-1];   // Estado después de empujar

        // 1. Identificar qué caja se movió comparando los sets
        point box_from = point(-1, -1);
        for (auto p : a.box_list) {
            if (b.box_list.find(p) == b.box_list.end()) {
                box_from = p; 
                break;
            }
        }
        point box_to = point(-1, -1);
        for (auto p : b.box_list) {
            if (a.box_list.find(p) == a.box_list.end()) {
                box_to = p; 
                break;
            }
        }

        // 2. Determinar el vector de empuje (delta) y el caracter
        point delta = point(box_to.x - box_from.x, box_to.y - box_from.y);
        int dir_idx = -1;
        for (int d = 0; d < 4; d++) {
            if (dirs[d].x == delta.x && dirs[d].y == delta.y) {
                dir_idx = d; break;
            }
        }

        char push_char = push_chars[dir_idx];
        
        // La posición desde la cual el jugador debe empujar
        point player_target = point(box_from.x - delta.x, box_from.y - delta.y);

        // 3. Obtener matriz de colisiones para caminar
        std::vector<std::vector<char>> grid;
        a.get_matrix0(grid); // 'a' es el estado actual donde la caja todavía no se mueve

        // 4. BFS para calcular los movimientos del jugador hasta la posición de empuje
        std::string player_moves = "";
        if (!(a.person_point == player_target)) {
            std::queue<std::pair<point, std::string>> q;
            std::vector<std::vector<bool>> visited(grid.size(), std::vector<bool>(grid[0].size(), false));

            q.push({a.person_point, ""});
            visited[a.person_point.x][a.person_point.y] = true;

            while (!q.empty()) {
                auto curr = q.front().first;
                auto path = q.front().second;
                q.pop();

                if (curr == player_target) {
                    player_moves = path;
                    break;
                }

                for (int d = 0; d < 4; ++d) {
                    point next_p = point(curr.x + dirs[d].x, curr.y + dirs[d].y);
                    
                    if (next_p.x >= 0 && next_p.x < (int)grid.size() && 
                        next_p.y >= 0 && next_p.y < (int)grid[0].size()) {
                        
                        if (!visited[next_p.x][next_p.y] && grid[next_p.x][next_p.y] == constant::BLANK) {
                            visited[next_p.x][next_p.y] = true;
                            q.push({next_p, path + move_chars[d]});
                        }
                    }
                }
            }
        }

        // 5. Concatenar camino del jugador + empuje de caja
        full_path += player_moves + push_char;
    }

    return full_path;
}

SolverStats game_solver::test_template(
    Method input,
    Heuristic heuristic_type,
    std::vector<game_node>& solution,
    bool calc_path_branching
) {
    // 1. INICIAMOS EL TIMER AQUÍ (Contabiliza el setup de la heurística y la memoria)
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = t_start; // Se sobreescribirá en el instante exacto del éxito
    bool goal_found = false;

    auto vec = Astar_init();
    vars_init(init);

    auto heuristic = [&](const game_node* a, const game_node*) {
        if (heuristic_type == Heuristic::hungarian) {
            int num_boxes = (int)a->box_list.size();
            int num_goals = (int)goal_positions.size();
            int sz = std::max(num_boxes, num_goals);

            vector<vector<int>> cost(sz, vector<int>(sz, 0));
            int i = 0;
            for (auto it = a->box_list.begin(); it != a->box_list.end(); ++it, ++i)
                for (int j = 0; j < num_goals; j++)
                    cost[i][j] = dist_to_goal[j][it->x][it->y];

            Hungarian h(cost);
            int min_cost = h.solve();

            int penalty_cost = penalty_solver.calculate_penalty(a->box_list);
            return min_cost >= 1000 ? min_cost : min_cost + penalty_cost;
        }
        else {
            int f = 0;
            for (auto i = a->box_list.begin(); i != a->box_list.end(); i++)
                f += vec[i->x][i->y];
            return f;
        }
    };

    // 2. EL CRONÓMETRO ESPÍA: Detiene el tiempo en el instante de la victoria
    is_equal = [&t_end, &goal_found](const game_node* a, const game_node*) -> bool {
        if (a->game_over()) {
            if (!goal_found) { // Capturar solo la primera vez que se toca la meta
                t_end = std::chrono::high_resolution_clock::now();
                goal_found = true;
            }
            return true;
        }
        return false;
    };

    Solver_template<vector<game_node>, game_node, Method::a_star> gsolver0;
    Solver_template<vector<game_node>, game_node, Method::dfs> gsolver1;
    Solver_template<vector<game_node>, game_node, Method::bfs> gsolver2;

    if (input == Method::a_star) {
        solution = gsolver0.solve(&init, nullptr, get_neighbors, is_visited, mark_visited, is_equal, heuristic);
    }
    else if (input == Method::dfs) {
        solution = gsolver1.solve(&init, nullptr, get_neighbors, is_visited, mark_visited, is_equal);
    }
    else if (input == Method::bfs) {
        solution = gsolver2.solve(&init, nullptr, get_neighbors, is_visited, mark_visited, is_equal);
    }

    // 3. Si no lo resolvió (Timeout o Unsolvable), el tiempo total es hasta este punto
    if (!goal_found) {
        t_end = std::chrono::high_resolution_clock::now();
    }

    SolverStats stats;

    // 4. GUARDAR EN MILISEGUNDOS EXACTOS
    stats.runtime_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    stats.pushes = solution.size();
    stats.generated_states = rpt.zobrist_hash.size();

    stats.expanded_nodes     = stat_expanded_nodes;
    stats.total_children     = stat_total_children;
    stats.effective_children = stat_effective_children;
    stats.repeated_nodes     = stat_repeated_nodes;
    stats.deadlocks          = stat_deadlocks;
    stats.closed_list_length = rpt.zobrist_hash.size();

    if (stat_expanded_nodes > 0) {
        stats.branching_real      = (double)stat_total_children     / stat_expanded_nodes;
        stats.branching_effective = (double)stat_effective_children / stat_expanded_nodes;
    }

    int depth = (int)solution.size();
    if (depth > 0) {
        stats.branching_classic = std::pow((double)stat_expanded_nodes, 1.0 / depth);
    }

    stats.redundancy = (stat_repeated_nodes > 0)
        ? (double)stat_total_children / stat_repeated_nodes
        : 0.0;

    bool timed_out =
        (input == Method::a_star && gsolver0.did_timeout()) ||
        (input == Method::dfs    && gsolver1.did_timeout()) ||
        (input == Method::bfs    && gsolver2.did_timeout());

    // Limpieza de Nodos Huérfanos
    if (input == Method::a_star) {
        for (const game_node* n : gsolver0.orphan_nodes) {
            if (n != nullptr && rpt.zobrist_hash.count(n) == 0) { 
                n->~game_node();
                game_mem.deallocate(const_cast<game_node*>(n));
            }
        }
        gsolver0.orphan_nodes.clear(); 
    }

    // Evaluación del Estado Final
    if (timed_out) {
        stats.status = SolveStatus::TIMEOUT;
    } else if (solution.empty()) {
        stats.status = SolveStatus::UNSOLVABLE;
    } else {
        stats.status = SolveStatus::SOLVED;
        stats.lurd_path = reconstruct_lurd(solution);
        
        // Contabilizar movimientos totales
        stats.moves = stats.lurd_path.length(); 
        
        if (calc_path_branching && !stats.lurd_path.empty()) {
            stats.path_stats = PathSimulator::compute_stats(original_map_1d, m, n, stats.lurd_path);
            stats.path_stats_calculated = true;
        }
    }

    vars_clear(init);

    return stats;
}

void print_solver_stats(const SolverStats& stats) {
    std::cout << "\n=========================================\n";
    std::cout << "        DUMP COMPLETO DE STATS           \n";
    std::cout << "=========================================\n";

    std::cout << "[STATUS Y SOLUCION]\n";
    std::cout << "status:                  "
              << (stats.status == SolveStatus::SOLVED    ? "SOLVED"     :
                  stats.status == SolveStatus::TIMEOUT   ? "TIMEOUT"    :
                                                           "UNSOLVABLE")
              << "\n";
    std::cout << "lurd_path:               " << stats.lurd_path << "\n";
    std::cout << "runtime_ms:              " << stats.runtime_ms << "\n";
    std::cout << "pushes:                  " << stats.pushes << "\n";
    std::cout << "moves (LURD length):     " << stats.moves << "\n";
    
    std::cout << "\n[ESTADISTICAS DE BUSQUEDA A*]\n";
    std::cout << "generated_states:        " << stats.generated_states << "\n";
    std::cout << "expanded_nodes:          " << stats.expanded_nodes << "\n";
    std::cout << "total_children:          " << stats.total_children << "\n";
    std::cout << "effective_children:      " << stats.effective_children << "\n";
    std::cout << "repeated_nodes:          " << stats.repeated_nodes << "\n";
    std::cout << "deadlocks:               " << stats.deadlocks << "\n";
    std::cout << "branching_real:          " << stats.branching_real << "\n";
    std::cout << "branching_effective:     " << stats.branching_effective << "\n";
    std::cout << "branching_classic:       " << stats.branching_classic << "\n";
    std::cout << "redundancy:              " << stats.redundancy << "\n";
    std::cout << "closed_list_length:      " << stats.closed_list_length << "\n";

    std::cout << "\n[ESTADISTICAS CALCULADAS DESDE LURD (SIMULADOR)]\n";
    std::cout << "path_stats_calculated:   " << (stats.path_stats_calculated ? "true" : "false") << "\n";

    if (stats.path_stats_calculated) {
        std::cout << "states (pasos en path):  " << stats.path_stats.states << "\n";
        
        std::cout << "box_lines:                       " << stats.path_stats.box_lines << "\n";
        std::cout << "box_changes:                     " << stats.path_stats.box_changes << "\n";
        
        std::cout << "branching_real_total_nodes:      " << stats.path_stats.branching_real_total_nodes << "\n";
        std::cout << "branching_real_min:              " << stats.path_stats.branching_real_min << "\n";
        std::cout << "branching_real_max:              " << stats.path_stats.branching_real_max << "\n";
        std::cout << "branching_real_avg:              " << stats.path_stats.get_branching_real_avg() << "\n";
        
        std::cout << "branching_effective_total_nodes: " << stats.path_stats.branching_effective_total_nodes << "\n";
        std::cout << "branching_effective_min:         " << stats.path_stats.branching_effective_min << "\n";
        std::cout << "branching_effective_max:         " << stats.path_stats.branching_effective_max << "\n";
        std::cout << "branching_effective_avg:         " << stats.path_stats.get_branching_effective_avg() << "\n";

        std::cout << "total_children_generated:        " << stats.path_stats.total_children_generated << "\n";
        std::cout << "repeated_nodes:                  " << stats.path_stats.repeated_nodes << "\n";
        std::cout << "deadlocks:                       " << stats.path_stats.deadlocks << "\n";
        std::cout << "redundancy:                      " << stats.path_stats.get_redundancy() << "\n";
    }
    std::cout << "=========================================\n";
}

// ==============================================================================
// DETECCIÓN DE BIPARTITE DEADLOCK (ALGORITMO HÚNGARO)
// ==============================================================================
bool game_solver::is_bipartite_deadlock(const game_node& node) {
    int num_boxes = (int)node.box_list.size();
    int num_goals = (int)goal_positions.size();
    
    // Matriz cuadrada requerida por el Algoritmo Húngaro
    int sz = std::max(num_boxes, num_goals);
    vector<vector<int>> cost(sz, vector<int>(sz, 0));

    int i = 0;
    for (auto it = node.box_list.begin(); it != node.box_list.end(); ++it, ++i) {
        for (int j = 0; j < num_goals; j++) {
            // Utilizamos la tabla de distancias invertidas que ya precalculaste
            cost[i][j] = dist_to_goal[j][it->x][it->y];
        }
    }

    Hungarian h(cost);
    double min_cost = h.solve();

    // Si el húngaro devuelve un coste altísimo (1000 es el INF de tu dist_to_goal),
    // es físicamente imposible que todas las cajas lleguen a una meta distinta.
    return min_cost >= 1000.0;
}