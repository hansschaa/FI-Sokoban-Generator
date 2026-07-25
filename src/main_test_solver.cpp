#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <torch/torch.h>

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
    size_t max_len = 0;

    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        max_len = std::max(max_len, line.length());
        board.emplace_back(line.begin(), line.end());
    }

    for (auto& row : board) {
        while (row.size() < max_len) {
            row.push_back(' ');
        }
    }

    return board;
}

int main(int argc, char* argv[])
{
    torch::set_num_threads(1);
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
    else if (heuristic_arg == "neural") {
        heuristic = Heuristic::neural_batched; // Default to batched for max speed
        std::cout << "Heuristica: Neural (Batched)\n";
    }
    else if (heuristic_arg == "neural_sequential") {
        heuristic = Heuristic::neural; // Sequential mode for Ablation Study
        std::cout << "Heuristica: Neural (Sequential)\n";
    }
    else if (heuristic_arg == "neural_batched") {
        heuristic = Heuristic::neural_batched;
        std::cout << "Heuristica: Neural Batched (Child Batching)\n";
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

    print_solver_stats(stats);
    
    return 0;
}