#include "../../../include/evolution/mutations/move_mutation.h"
#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"
#include "../../../include/game_solver.h"

#include <cstdlib>
#include <iostream>

std::vector<Pair> MoveMutation::getPositions(
    const std::vector<std::vector<char>>& board,
    char c)
{
    std::vector<Pair> positions;

    for (size_t i = 0; i < board.size(); i++)
    {
        for (size_t j = 0; j < board[i].size(); j++)
        {
            if (board[i][j] == c)
            {
                positions.push_back(
                    {(int)i, (int)j});
            }
        }
    }

    return positions;
}

Pair MoveMutation::getRandomEmpty(
    const std::vector<std::vector<char>>& board)
{
    std::vector<Pair> empty;

    for (size_t i = 0; i < board.size(); i++)
    {
        for (size_t j = 0; j < board[i].size(); j++)
        {
            if (board[i][j] == ' ')
            {
                empty.push_back(
                    {(int)i, (int)j});
            }
        }
    }

    return empty[rand() % empty.size()];
}

void MoveMutation::moveCharacter(
    std::vector<std::vector<char>>& board,
    char target)
{
    auto positions =
        getPositions(board, target);

    if (positions.empty())
        return;

    Pair selected =
        positions[rand() % positions.size()];

    Pair empty =
        getRandomEmpty(board);

    //
    // REMOVE OLD
    //

    board[selected.i][selected.j] = ' ';

    //
    // PLACE NEW
    //

    board[empty.i][empty.j] =
        target;
}

void MoveMutation::apply(Individual& ind)
{
    //
    // KEEP ORIGINAL
    //

    auto original =
        ind.board;

    //
    // RANDOM TYPE
    // 0 = player
    // 1 = box
    // 2 = goal
    //

    int type =
        rand() % 3;

    if (type == 0)
    {
        moveCharacter(ind.board, '@');
    }
    else if (type == 1)
    {
        moveCharacter(ind.board, '$');
    }
    else
    {
        moveCharacter(ind.board, '.');
    }

    //
    // VALIDATE WITH SOLVER
    //

    std::string level =
        board_to_string(ind.board);

    game_solver solver(
        level,
        ind.board.size(),
        ind.board[0].size(),
        512);

    std::vector<game_node> solution;

    auto stats =
        solver.test_template(1, solution);

    //
    // REJECT INVALID
    //

    if (stats.status != SolveStatus::SOLVED)
    {
        ind.board = original;
    }
    else
    {
        ind.fitness =
            stats.pushes;
    }
}