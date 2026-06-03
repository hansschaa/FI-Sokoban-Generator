#include "../../../include/evolution/mutations/add_mutation.h"

#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"

#include <vector>
#include <cstdlib>
#include <algorithm>

bool AddMutation::apply(
    Individual& individual)
{
    auto& b = individual.board;

    //
    // DYNAMIC BOX LIMIT
    // max_boxes scales with the navigable space of the current board.
    // 1 box per 15 free cells, clamped to [3, 6].
    // Minimum of 3 ensures small shells aren't trivially easy.
    // Maximum of 6 prevents overcrowding on large shells.
    //

    const int free_cells = count_free_cells(b);
    const int max_boxes  = std::max(3, std::min(6, free_cells / 15));

    if (count_boxes(b) >= max_boxes)
        return false;

    //
    // FIND EMPTY CELLS
    //

    std::vector<Pair> emptyCells;

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