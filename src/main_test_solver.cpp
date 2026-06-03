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
        std::cerr << "Uso: ./test_solver <ruta/al/level.txt> <heuristica>\n"
                  << "  heuristica: simple | hungarian\n"
                  << "Ejemplo: ./test_solver ../levels/level1.txt hungarian\n";
        return 1;
    }

    //
    // LEER TABLERO DESDE ARCHIVO
    //

    std::vector<std::vector<char>> board;

    try
    {
        board = load_board(argv[1]);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    //
    // PARSEAR HEURISTICA
    //

    std::string heuristic_arg = argv[2];
    Heuristic heuristic;

    if (heuristic_arg == "hungarian")
    {
        heuristic = Heuristic::hungarian;
        std::cout << "Heuristica: Hungarian (asignacion optima caja->objetivo)\n";
    }
    else if (heuristic_arg == "simple")
    {
        heuristic = Heuristic::simple;
        std::cout << "Heuristica: Simple (suma individual)\n";
    }
    else
    {
        std::cerr << "Heuristica desconocida: \"" << heuristic_arg << "\"\n"
                  << "Opciones validas: simple | hungarian\n";
        return 1;
    }

    //
    // MOSTRAR TABLERO
    //

    std::cout << "TABLERO:\n";

    for (const auto& row : board)
    {
        for (char c : row)
            std::cout << c;

        std::cout << '\n';
    }

    std::cout << '\n';

    //
    // CONVERTIR A STRING
    //

    std::string level = board_to_string(board);

    unsigned int rows = board.size();
    unsigned int cols = board.empty() ? 0 : board[0].size();

    //
    // CREAR SOLVER
    //

    game_solver solver(level, rows, cols, 512);

    //
    // RESOLVER CON A*
    //

    std::vector<game_node> solution;

    std::cout << "Resolviendo con A*...\n\n";

    auto stats = solver.test_template(Method::a_star, heuristic, solution);

    //
    // MOSTRAR STATS
    //

    std::cout << "\n--- RESULTADO ---\n";

    std::cout << "status:               "
              << (stats.status == SolveStatus::SOLVED    ? "SOLVED"     :
                  stats.status == SolveStatus::TIMEOUT   ? "TIMEOUT"    :
                                                           "UNSOLVABLE")
              << "\n";

    std::cout << "pushes:               " << stats.pushes             << "\n";
    std::cout << "runtime_sec:          " << stats.runtime_sec        << "\n";
    std::cout << "generated_states:     " << stats.generated_states   << "\n";
    std::cout << "expanded_nodes:       " << stats.expanded_nodes      << "\n";
    std::cout << "total_children:       " << stats.total_children      << "\n";
    std::cout << "effective_children:   " << stats.effective_children  << "\n";
    std::cout << "repeated_nodes:       " << stats.repeated_nodes      << "\n";
    std::cout << "deadlocks:            " << stats.deadlocks           << "\n";
    std::cout << "branching_real:       " << stats.branching_real      << "\n";
    std::cout << "branching_effective:  " << stats.branching_effective << "\n";
    std::cout << "branching_classic:    " << stats.branching_classic   << "\n";
    std::cout << "redundancy:           " << stats.redundancy          << "\n";
    std::cout << "closed_list_length:   " << stats.closed_list_length  << "\n";
    std::cout << "solution_nodes:       " << solution.size()           << "\n";
    std::cout << "----------------------\n";

    return 0;
}