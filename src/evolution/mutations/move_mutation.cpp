#include "../../../include/evolution/mutations/move_mutation.h"

#include "../../../include/evolution/utils/pair.h"
#include "../../../include/evolution/utils/board_utils.h"

#include <cstdlib>
#include <iostream>

std::vector<Pair> MoveMutation::getPositions(
    const std::vector<std::vector<char>>& board,
    char target)
{
    std::vector<Pair> positions;

    for (size_t i = 0; i < board.size(); i++)
    {
        for (size_t j = 0; j < board[i].size(); j++)
        {
            char c = board[i][j];

            if (target == '@')
            {
                if (c == '@' || c == '+')
                    positions.push_back({(int)i, (int)j});
            }
            else if (target == '$')
            {
                if (c == '$' || c == '*')
                    positions.push_back({(int)i, (int)j});
            }
            else if (target == '.')
            {
                if (c == '.' || c == '+' || c == '*')
                    positions.push_back({(int)i, (int)j});
            }
        }
    }

    return positions;
}

//
// GET RANDOM CELL VALID FOR TARGET
// @ and $ can move to ' ' or '.'
// . can move to ' ', '@', '$'
//

Pair MoveMutation::getRandomDestination(
    const std::vector<std::vector<char>>& board,
    char target)
{
    std::vector<Pair> candidates;

    for (size_t i = 0; i < board.size(); i++)
    {
        for (size_t j = 0; j < board[i].size(); j++)
        {
            char c = board[i][j];

            if (target == '@' || target == '$')
            {
                //
                // CAN LAND ON EMPTY OR GOAL
                //
                
                if (c == ' ' || c == '.') {
                    if (target == '$' && !deadlock_mask.empty() && i < deadlock_mask.size() && j < deadlock_mask[i].size() && deadlock_mask[i][j]) {
                        continue; // Skip deadlock cells for boxes
                    }
                    candidates.push_back({(int)i, (int)j});
                }
            }
            else if (target == '.')
            {
                //
                // CAN LAND ON EMPTY, PLAYER, OR BOX
                //

                if (c == ' ' || c == '@' || c == '$')
                    candidates.push_back({(int)i, (int)j});
            }
        }
    }

    if (candidates.empty())
        return {-1, -1};

    return candidates[rand() % candidates.size()];
}

bool MoveMutation::moveCharacter(
    std::vector<std::vector<char>>& board,
    char target)
{
    auto positions = getPositions(board, target);

    if (positions.empty())
        return false;

    Pair selected =
        positions[rand() % positions.size()];

    Pair dest =
        getRandomDestination(board, target);

    if (dest.i == -1)
        return false;

    //
    // SAME CELL → NO-OP
    //

    if (selected.i == dest.i &&
        selected.j == dest.j)
        return false;

    char src = board[selected.i][selected.j];
    char dst = board[dest.i][dest.j];

    //
    // CLEAR SOURCE CELL
    // Handle composite states correctly depending on what is moving
    //

    if (src == '+')
    {
        if (target == '.') board[selected.i][selected.j] = '@'; // moving goal, player remains
        else               board[selected.i][selected.j] = '.'; // moving player, goal remains
    }
    else if (src == '*')
    {
        if (target == '.') board[selected.i][selected.j] = '$'; // moving goal, box remains
        else               board[selected.i][selected.j] = '.'; // moving box, goal remains
    }
    else
    {
        board[selected.i][selected.j] = ' ';
    }

    //
    // PLACE ENTITY AT DESTINATION
    // Handle composite states correctly
    //

    if (target == '@')
    {
        board[dest.i][dest.j] =
            (dst == '.') ? '+' : '@';
    }
    else if (target == '$')
    {
        board[dest.i][dest.j] =
            (dst == '.') ? '*' : '$';
    }
    else if (target == '.')
    {
        if      (dst == '@') board[dest.i][dest.j] = '+';
        else if (dst == '$') board[dest.i][dest.j] = '*';
        else                 board[dest.i][dest.j] = '.';
    }

    return true;
}

bool MoveMutation::apply(
    Individual& ind)
{
    int type = rand() % 3;

    bool changed = false;

    if (type == 0)
        changed = moveCharacter(ind.board, '@');
    else if (type == 1)
        changed = moveCharacter(ind.board, '$');
    else
        changed = moveCharacter(ind.board, '.');

    //
    // STRUCTURAL VALIDATION ONLY
    // Solvability is checked by the evaluator
    //

    return changed;
}