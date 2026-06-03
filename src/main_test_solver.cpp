#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "../include/game_solver.h"
#include "../include/evolution/utils/board_utils.h"

std::vector<std::vector<char>> load_board(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("No se pudo abrir: " + filename);
    }

    std::vector<std::vector<char>> board;
    std::string line;

    while (std::getline(file, line))
    {
        board.emplace_back(line.begin(), line.end());
    }

    return board;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Uso: ./test_solver <ruta/al/level.txt> <heuristica> [calc]\n"
                  << "  heuristica: simple | hungarian\n"
                  << "  calc: opcional, pon 'calc' para simular Path Branching Stats\n"
                  << "Ejemplo: ./test_solver ../levels/level1.txt hungarian calc\n";
        return 1;
    }

    std::vector<std::vector<char>> board;

    try {
        board = load_board(argv[1]);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::string heuristic_arg = argv[2];
    Heuristic heuristic;

    if (heuristic_arg == "hungarian") {
        heuristic = Heuristic::hungarian;
        std::cout << "Heuristica: Hungarian\n";
    }
    else if (heuristic_arg == "simple") {
        heuristic = Heuristic::simple;
        std::cout << "Heuristica: Simple\n";
    }
    else {
        std::cerr << "Heuristica desconocida: \"" << heuristic_arg << "\"\n";
        return 1;
    }

    // AÑADIDO: Leer flag opcional para branch simulator
    bool calc_path_branching = false;
    if (argc >= 4) {
        std::string flag = argv[3];
        if (flag == "calc" || flag == "1" || flag == "true") {
            calc_path_branching = true;
            std::cout << "Path Branching Simulator: ACTIVADO\n";
        }
    }

    std::cout << "TABLERO:\n";
    for (const auto& row : board) {
        for (char c : row) std::cout << c;
        std::cout << '\n';
    }
    std::cout << '\n';

    std::string level = board_to_string(board);
    unsigned int rows = board.size();
    unsigned int cols = board.empty() ? 0 : board[0].size();

    game_solver solver(level, rows, cols, 512);
    std::vector<game_node> solution;

    std::cout << "Resolviendo con A*...\n\n";

    auto stats = solver.test_template(Method::a_star, heuristic, solution, calc_path_branching);

    //
    // MOSTRAR DUMP COMPLETO DE STATS
    //

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
    std::cout << "runtime_sec:             " << stats.runtime_sec << "\n";
    std::cout << "pushes:                  " << stats.pushes << "\n";
    

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

    return 0;
}