#include "../../include/evolution/evaluator.h"

#include "../../include/game_solver.h"
#include "../../include/game_node.h"

#include "../../include/evolution/utils/board_utils.h"

#include <iostream>

double Evaluator::evaluate(Individual& ind)
{
    std::string level =
        board_to_string(ind.board);

    std::cout << "LEVEL:\n";
    std::cout << level << std::endl;

    unsigned int rows =
    ind.board.size();

    unsigned int cols =
        ind.board[0].size();

    game_solver solver(
        level,
        rows,
        cols,
        64
    );

    std::vector<game_node> solution;

    SolverStats stats =
        solver.test_template(0, solution);

    ind.pushes = stats.pushes;

    ind.explored =
        stats.explored_states;

    ind.solved =
        stats.status == SolveStatus::SOLVED;

    if (!ind.solved) {

        ind.fitness = -1e9;
        return ind.fitness;
    }

    ind.fitness = stats.pushes;

    return ind.fitness;
}