#include "../../../include/evolution/evaluator/evaluator.h"
#include "../../../include/game_solver.h"
#include "../../../include/evolution/utils/board_utils.h"
#include <cmath>
#include <vector>

double Evaluator::evaluate(
    Individual& individual)
{
    //
    // BOARD → STRING
    //
    std::string level =
        board_to_string(individual.board);

    unsigned int rows =
        individual.board.size();

    unsigned int cols =
        individual.board[0].size();

    //
    // SOLVER
    //
    game_solver solver(
        level,
        rows,
        cols,
        512);

    std::vector<game_node> solution;

    auto stats =
        solver.test_template(
            Method::a_star,
            solution);

    //
    // UNSOLVABLE
    //
    if (stats.status !=
        SolveStatus::SOLVED)
    {
        individual.fitness = -1e9;
        return individual.fitness;
    }

    //
    // FITNESS
    //
    switch (fitnessType)
    {
        case FitnessType::PUSHES:
            individual.fitness =
                stats.pushes;
            break;
        case FitnessType::EXPANDED_NODES:
            individual.fitness =
                stats.generated_states;
            break;
        case FitnessType::SOLUTION_LENGTH:
            individual.fitness =
                solution.size();
            break;
        case FitnessType::EFFECTIVE_BRANCHING_FACTOR:
            individual.fitness =
                computeEffectiveBranchingFactor(
                    stats.generated_states,
                    solution.size());
            break;
    }

    return individual.fitness;
}

double Evaluator::computeEffectiveBranchingFactor(
    double expanded,
    double depth)
{
    //
    // SIMPLE APPROXIMATION
    //
    // expanded ≈ 1 + b + b² + ... + b^d
    //
    if (depth <= 1)
    {
        return -1e9;
    }

    if (expanded <= 0)
    {
        return -1e9;
    }

    double b =
        pow(expanded, 1.0 / depth);

    return b;
}