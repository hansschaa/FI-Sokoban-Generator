#pragma once

#include "game_node.h"
#include "repeat.h"
#include "locked.h"
#include "my_memory.h"
#include "method.h"        
#include "path_simulator.h" // AÑADIDO: Simulador de Branching
#include "penalty.h"
#include <string>
#include <functional>
#include <vector>
#include <memory>

class NeuralHeuristic;

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

    std::string lurd_path;

    double runtime_ms = 0.0;

    int pushes = 0;
    int moves = 0;

    size_t generated_states = 0;

    // Estadísticas de búsqueda originales A*
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

    // AÑADIDO: Estadísticas calculadas desde la solución LURD
    bool path_stats_calculated = false;
    PathBranchingStats path_stats;
    
    // AÑADIDO: Distancia Bipartita Inicial
    double initial_optimal_distance = 0.0;
};

// AÑADIDO: Declaración de la función global para imprimir las stats
void print_solver_stats(const SolverStats& stats);


enum class Heuristic {
    simple,    
    manhattan,
    hungarian,
    neural,
    neural_batched,         // batch de hijos de UN nodo (2-12 por llamada GPU)
    neural_batched_massive  // cross-node batch (BATCH_K=64, ~400 por llamada GPU)
};

class game_solver {

private:
    game_node init;
    repeat rpt;
    locked lk;
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

    // Guardar mapa original para el simulador
    std::string original_map_1d;

    long stat_expanded_nodes     = 0;
    long stat_total_children     = 0;
    long stat_effective_children = 0;
    long stat_repeated_nodes     = 0;
    long stat_deadlocks          = 0;

    std::vector<point> goal_positions;
    std::vector<std::vector<std::vector<int>>> dist_to_goal;

    // --- NUEVO: Método Bipartito ---
    bool is_bipartite_deadlock(const game_node& node);

    // --- NUEVO: Resolvedor de Penalizaciones ---
    Penalty penalty_solver;

public:
    game_solver(std::string& game_map, unsigned int mm, unsigned int nn, int memval);

    // --- NUEVO: FLAG METODOLÓGICO PARA EXPERIMENTOS ---
    // true: Poda agresiva (Súper rápido, ideal para FO1, FO2, FO3)
    // false: Solo poda básica (Lento, obligatorio para contar FO4 Deadlocks)
    bool enable_advanced_deadlocks = false;
    
    SolverStats test_template(Method method, Heuristic heuristic, std::vector<game_node>& solution, bool calc_path_branching = false, std::shared_ptr<NeuralHeuristic> external_net = nullptr);

    SolverStats test_template(Method method, std::vector<game_node>& solution, bool calc_path_branching = false, std::shared_ptr<NeuralHeuristic> external_net = nullptr) {
        return test_template(method, Heuristic::simple, solution, calc_path_branching, external_net);
    }
};