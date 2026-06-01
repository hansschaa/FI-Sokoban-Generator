#include "../../../include/evolution/mutations/add_mutation.h"

#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"

#include <vector>
#include <cstdlib>

bool AddMutation::apply(
    Individual& individual)
{
    auto& b = individual.board;

    std::vector<Pair> emptyCells;

    //
    // FIND EMPTY CELLS
    //

    for (int i = 0; i < (int)b.size(); i++)
        for (int j = 0; j < (int)b[i].size(); j++)
            if (b[i][j] == ' ')
                emptyCells.push_back({i, j});

    //
    // NEED AT LEAST 2 EMPTY CELLS
    //

    if (emptyCells.size() < 2)
        return false;

    //
    // PLACE RANDOM BOX
    //

    int idx1 = rand() % emptyCells.size();
    Pair p1  = emptyCells[idx1];
    b[p1.i][p1.j] = '$';

    emptyCells.erase(emptyCells.begin() + idx1);

    //
    // PLACE RANDOM GOAL
    //

    int idx2 = rand() % emptyCells.size();
    Pair p2  = emptyCells[idx2];
    b[p2.i][p2.j] = '.';

    //
    // STRUCTURAL VALIDATION ONLY
    // Solvability is checked by the evaluator
    //

    return true;
}