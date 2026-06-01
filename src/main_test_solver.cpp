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

int main()
{
    //
    // LEER TABLERO DESDE ARCHIVO
    //

    std::vector<std::vector<char>> board;

    try
    {
        board = load_board("level.txt");
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
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

    auto stats = solver.test_template(Method::a_star, solution);

    //
    // MOSTRAR STATS
    //

    std::cout << "--- RESULTADO ---\n";
    std::cout << "status:          "
              << (stats.status == SolveStatus::SOLVED ? "SOLVED" : "UNSOLVABLE")
              << "\n";

    std::cout << "pushes:          " << stats.pushes << "\n";
    std::cout << "explored_states: " << stats.explored_states << "\n";
    std::cout << "runtime_sec:     " << stats.runtime_sec << "\n";
    std::cout << "pasos solucion:  " << solution.size() << "\n";
    std::cout << "-----------------\n";

    return 0;
}