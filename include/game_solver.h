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
    double runtime_sec;
    int pushes;
    int generated_states;
    int explored_states;   // FIX: faltaba; usado en evaluator.cpp y main_benchmark.cpp
    SolveStatus status;
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

    int get_nums2(const game_node& input);   // FIX: const ref, evita copiar set<point>
    void set_lambda_function();
    std::vector<point> get_legal_point(std::vector<std::vector<char>>& vec, point p);
    std::vector<std::vector<int>> Astar_init();
    void vars_init(game_node& input);
    void vars_clear(game_node& input);

public:
    game_solver(std::string& game_map, unsigned int mm, unsigned int nn, int memval);
    SolverStats test_template(Method input, std::vector<game_node>& solution);  // FIX: Method en lugar de int
};