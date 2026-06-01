#pragma once

#include "game_node.h"
#include "repeat.h"
#include "locked.h"
#include "my_memory.h"
#include "method.h"        // FIX: Method antes que cualquier uso suyo
#include <string>
#include <functional>
#include <vector>

class detect_legal {
public:
    std::vector<std::vector<bool>> value;
    std::vector<std::vector<char>> matrix_with_box;
    detect_legal(const game_node* node);
    detect_legal(std::vector<std::vector<char>>& matrix, point& start);
    detect_legal();
    bool can_get(point& des);
    bool can_box_move(point& box, point& person);
};

enum class SolveStatus {
    SOLVED,
    UNSOLVABLE,
    TIMEOUT
};

struct SolverStats {

    SolveStatus status;

    double runtime_sec = 0.0;

    int pushes = 0;

    size_t generated_states = 0;

    // Estadísticas de búsqueda

    long expanded_nodes = 0;

    long total_children = 0;
    long effective_children = 0;

    long repeated_nodes = 0;
    long deadlocks = 0;

    double branching_real = 0.0;
    double branching_effective = 0.0;
    double branching_classic = 0.0;

    double redundancy = 0.0;

    size_t closed_list_length = 0;
};

enum class Heuristic {
    simple,    // suma individual vec[box.x][box.y] (original)
    hungarian  // asignación óptima caja→objetivo (lower bound real)
};

class game_solver {

private:
    game_node init;
    repeat rpt;
    locked lk;
    my_memory_pool game_mem;
    std::function<void(const game_node*, std::function<void(const game_node*)>)> get_neighbors;
    std::function<bool(const game_node*)> is_visited;
    std::function<void(const game_node*)> mark_visited;
    std::function<bool(const game_node*, const game_node*)> is_equal;

    int get_nums2(const game_node& input);
    void set_lambda_function();
    std::vector<point> get_legal_point(std::vector<std::vector<char>>& vec, point p);
    std::vector<std::vector<int>> Astar_init();
    void vars_init(game_node& input);
    void vars_clear(game_node& input);

    // Contadores que se llenan durante la búsqueda (equivalentes a movesHistory de Java)
    long stat_expanded_nodes     = 0;
    long stat_total_children     = 0;
    long stat_effective_children = 0;
    long stat_repeated_nodes     = 0;
    long stat_deadlocks          = 0;

    // Posiciones de objetivos, cacheadas en Astar_init() para la heurística Hungarian
    std::vector<point> goal_positions;

    // dist_to_goal[g][x][y] = pushes mínimos de una caja en (x,y) al objetivo g
    std::vector<std::vector<std::vector<int>>> dist_to_goal;

public:
    game_solver(std::string& game_map, unsigned int mm, unsigned int nn, int memval);
    SolverStats test_template(Method method, Heuristic heuristic, std::vector<game_node>& solution);

    // Sobrecarga para compatibilidad con código existente — usa heurística simple
    SolverStats test_template(Method method, std::vector<game_node>& solution) {
        return test_template(method, Heuristic::simple, solution);
    }
};