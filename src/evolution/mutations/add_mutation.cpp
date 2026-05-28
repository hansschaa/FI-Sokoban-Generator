#include "../../../include/evolution/mutations/add_mutation.h"

#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"

#include "../../../include/game_solver.h"

#include <vector>
#include <cstdlib>

void AddMutation::apply(
    Individual& individual)
{
    //
    // BACKUP ORIGINAL BOARD
    //

    auto original =
        individual.board;

    auto& b =
        individual.board;

    std::vector<Pair> emptyCells;

    //
    // FIND EMPTY CELLS
    //

    for (int i = 0; i < (int)b.size(); i++)
    {
        for (int j = 0; j < (int)b[i].size(); j++)
        {
            if (b[i][j] == ' ')
            {
                emptyCells.push_back({i, j});
            }
        }
    }

    //
    // NEED 2 CELLS
    //

    if (emptyCells.size() < 2)
    {
        return;
    }

    //
    // RANDOM BOX
    //

    int idx1 =
        rand() % emptyCells.size();

    Pair p1 =
        emptyCells[idx1];

    b[p1.i][p1.j] = '$';

    //
    // REMOVE USED CELL
    //

    emptyCells.erase(
        emptyCells.begin() + idx1);

    //
    // RANDOM GOAL
    //

    int idx2 =
        rand() % emptyCells.size();

    Pair p2 =
        emptyCells[idx2];

    b[p2.i][p2.j] = '.';

    //
    // VALIDATE SOLVABILITY
    //

    std::string level =
        board_to_string(b);

    unsigned int rows =
        b.size();

    unsigned int cols =
        b[0].size();

    game_solver solver(
        level,
        rows,
        cols,
        512);

    std::vector<game_node> solution;

    auto stats =
        solver.test_template(
            1,
            solution);

    //
    // REVERT IF UNSOLVABLE
    //

    if (stats.status !=
        SolveStatus::SOLVED)
    {
        individual.board =
            original;
    }
}