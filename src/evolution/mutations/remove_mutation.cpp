#include "../../../include/evolution/mutations/remove_mutation.h"

#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"

#include "../../../include/game_solver.h"

#include <vector>
#include <cstdlib>

void RemoveMutation::apply(
    Individual& individual)
{
    auto original =
        individual.board;

    auto& b =
        individual.board;

    std::vector<Pair> boxes;

    std::vector<Pair> goals;

    //
    // FIND BOXES + GOALS
    //

    for (int i = 0; i < (int)b.size(); i++)
    {
        for (int j = 0; j < (int)b[i].size(); j++)
        {
            if (b[i][j] == '$')
            {
                boxes.push_back({i, j});
            }

            else if (b[i][j] == '.')
            {
                goals.push_back({i, j});
            }
        }
    }

    //
    // KEEP AT LEAST 1 PAIR
    //

    if (boxes.size() <= 1 ||
        goals.size() <= 1)
    {
        return;
    }

    //
    // REMOVE RANDOM BOX
    //

    Pair bpos =
        boxes[rand() % boxes.size()];

    b[bpos.i][bpos.j] = ' ';

    //
    // REMOVE RANDOM GOAL
    //

    Pair gpos =
        goals[rand() % goals.size()];

    b[gpos.i][gpos.j] = ' ';

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