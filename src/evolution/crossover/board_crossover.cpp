#include "../../../include/evolution/crossover/board_crossover.h"

#include <cstdlib>

Individual BoardCrossover::apply(
    const Individual& parent1,
    const Individual& parent2)
{
    Individual child =
        parent1;

    int rows =
        child.board.size();

    int cols =
        child.board[0].size();

    //
    // RANDOM CUT ROW
    //

    int cut =
        rand() % rows;

    //
    // COPY LOWER PART
    // FROM PARENT2
    //

    for (int i = cut; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            child.board[i][j] =
                parent2.board[i][j];
        }
    }

    return child;
}