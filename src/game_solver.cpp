#include <deque>
#include <cstdio>
#include <cmath>
#include <chrono>
#include "game_solver.h"
#include "constant.h"
#include "locked.h"
#include "repeat.h"
#include "solver_template.h"
#include "mazesolver.h"

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
    //
    // CLEAR THE HASH SET
    // Avoids dangling pointers into the now-reset pool
    //
    rpt.zobrist_hash.clear();
    // FIX: antes no se llamaba game_mem.clear(), dejando el pool con bloques
    // marcados como usados. La siguiente llamada a vars_init + solve podía
    // quedarse sin bloques disponibles en el pool prematuramente.
    game_mem.clear();
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

vector<vector<int>> game_solver::Astar_init() {
    vector<vector<int>> result(m,vector<int>(n,0));

    for (int i = 0; i <m; i++) {
        for (int j = 0; j < n; j++) {
            if (blank_matrix[i][j] == WALL) {
                result[i][j] = 1000;
                continue;
            }
            game_node new_node;
            new_node.box_list.insert(point(i,j));
            vector<vector<char>>vec;
            new_node.get_matrix0(vec);
            auto person_point = get_legal_point(vec, point(i,j));
            if (person_point.empty()) {
                result[i][j] = 1000;
                continue;
            }
            // FIX: antes se llamaba get_matrix0(vec) aquí, con person_point
            // sin inicializar. Ahora se asigna person_point primero dentro
            // del loop, y get_matrix0 se llama solo cuando es necesario.
            int min_num = INT32_MAX;
            for (auto& dd : person_point){
                new_node.person_point = dd;
                // FIX: get_nums2 ahora recibe const ref, no copia
                int min_ = get_nums2(new_node);
                if (min_ < min_num) {
                    min_num = min_;
                }
            }
            result[i][j] = min_num;
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

                        if (lk.is_locked(new_box_point, temp_matrix2)) {
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
    };//todo:实际只用来判断终点，但终点和普通节点判断逻辑不同

}

SolverStats game_solver::test_template(
    Method input,   // FIX: antes era int con magic numbers 0/1/2 sin documentar
    std::vector<game_node>& solution
) {

    auto vec = Astar_init();

    vars_init(init);

    auto heuristic = [&](const game_node* a, const game_node*) {

        int f = 0;

        for (auto i = a->box_list.begin();
             i != a->box_list.end();
             i++) {

            auto p = *i;

            f += vec[p.x][p.y];
        }

        return f;
    };

    Solver_template<vector<game_node>, game_node, Method::a_star> gsolver0;
    Solver_template<vector<game_node>, game_node, Method::dfs> gsolver1;
    Solver_template<vector<game_node>, game_node, Method::bfs> gsolver2;

    printf("compute start!!\n");

    auto t1 = chrono::high_resolution_clock::now();

    if (input == Method::a_star)
    {
        solution = gsolver0.solve(
            &init,
            nullptr,
            get_neighbors,
            is_visited,
            mark_visited,
            is_equal,
            heuristic
        );
    }
    else if (input == Method::dfs)
    {
        solution = gsolver1.solve(
            &init,
            nullptr,
            get_neighbors,
            is_visited,
            mark_visited,
            is_equal
        );
    }
    else if (input == Method::bfs)
    {
        solution = gsolver2.solve(
            &init,
            nullptr,
            get_neighbors,
            is_visited,
            mark_visited,
            is_equal
        );
    }

    auto t2 = chrono::high_resolution_clock::now();

    printf("compute complete!!\n");

    SolverStats stats;

    stats.runtime_sec =
        chrono::duration<double>(t2 - t1).count();

    stats.pushes =
        solution.size();

    stats.generated_states =
        rpt.zobrist_hash.size();

    // Contadores directos (equivalentes directos de movesHistory en Java)
    stats.expanded_nodes     = stat_expanded_nodes;
    stats.total_children     = stat_total_children;
    stats.effective_children = stat_effective_children;
    stats.repeated_nodes     = stat_repeated_nodes;
    stats.deadlocks          = stat_deadlocks;
    stats.closed_list_length = rpt.zobrist_hash.size();

    // Métricas derivadas — mismo cálculo que Java en forwardSearch
    // branchingReal    = totalChildren        / expandedNodes
    // branchingEffective = totalEffectiveChildren / expandedNodes
    // branchingClassic = expandedNodes ^ (1 / depth)
    // redundancy       = totalChildren / repeatedNodes  (0 si no hubo repetidos)
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

    // FIX: antes se repetía if(input==0) ... elif(input==1) ... para did_timeout.
    // Ahora se consulta did_timeout() de cada solver solo si fue el que se usó,
    // evitando llamar did_timeout() sobre solvers nunca ejecutados.
    bool timed_out =
        (input == Method::a_star && gsolver0.did_timeout()) ||
        (input == Method::dfs    && gsolver1.did_timeout()) ||
        (input == Method::bfs    && gsolver2.did_timeout());

    if (timed_out) {
        stats.status = SolveStatus::TIMEOUT;
    } else if (solution.empty()) {
        stats.status = SolveStatus::UNSOLVABLE;
    } else {
        stats.status = SolveStatus::SOLVED;
    }

    vars_clear(init);

    return stats;
}