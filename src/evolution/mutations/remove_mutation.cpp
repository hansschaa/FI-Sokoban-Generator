#include "../../../include/evolution/mutations/remove_mutation.h"

#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"

#include <vector>
#include <cstdlib>

bool RemoveMutation::apply(
    Individual& individual)
{
    auto& b = individual.board;

    std::vector<Pair> boxes;
    std::vector<Pair> goals;

    //
    // FIND BOXES + GOALS
    //

    for (int i = 0; i < (int)b.size(); i++)
    {
        for (int j = 0; j < (int)b[i].size(); j++)
        {
            char c = b[i][j];

            if (c == '$' || c == '*')
                boxes.push_back({i, j});

            if (c == '.' || c == '+' || c == '*')
                goals.push_back({i, j});
        }
    }

    //
    // KEEP AT LEAST 1 PAIR
    //

    if (boxes.size() <= 1 ||
        goals.size() <= 1)
        return false;

    //
    // REMOVE RANDOM BOX
    //

    Pair bpos = boxes[rand() % boxes.size()];
    char bc   = b[bpos.i][bpos.j];

    if (bc == '*')
        b[bpos.i][bpos.j] = '.';   // box on goal → restore goal
    else
        b[bpos.i][bpos.j] = ' ';

    //
    // REMOVE RANDOM GOAL
    // Exclude the cell already modified above
    //

    std::vector<Pair> remainingGoals;

    for (auto& g : goals)
        if (!(g.i == bpos.i && g.j == bpos.j))
            remainingGoals.push_back(g);

    if (remainingGoals.empty())
    {
        //
        // ROLLBACK
        //

        b[bpos.i][bpos.j] = bc;
        return false;
    }

    Pair gpos = remainingGoals[rand() % remainingGoals.size()];
    char gc   = b[gpos.i][gpos.j];

    if      (gc == '+') b[gpos.i][gpos.j] = '@';   // player on goal → restore player
    else if (gc == '*') b[gpos.i][gpos.j] = '$';   // box on goal → restore box
    else                b[gpos.i][gpos.j] = ' ';

    //
    // STRUCTURAL VALIDATION ONLY
    // Solvability is checked by the evaluator
    //

    return true;
}