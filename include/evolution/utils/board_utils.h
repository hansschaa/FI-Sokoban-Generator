#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include "pair.h"

//
// FOR SOLVER
// NO NEWLINES
//

inline std::string board_to_string(
    const std::vector<std::vector<char>>& board)
{
    std::string result;

    for (const auto& row : board)
    {
        for (char c : row)
            result += c;
    }

    return result;
}

//
// FOR PRINTING
// WITH NEWLINES
//

inline std::string board_to_pretty_string(
    const std::vector<std::vector<char>>& board)
{
    std::string result;

    for (const auto& row : board)
    {
        for (char c : row)
            result += c;

        result += '\n';
    }

    return result;
}

//
// COUNT BOXES
//

inline int count_boxes(
    const std::vector<std::vector<char>>& board)
{
    int count = 0;

    for (const auto& row : board)
    {
        for (char c : row)
        {
            if (c == '$' || c == '*')
                count++;
        }
    }

    return count;
}

//
// COUNT FREE CELLS
// Counts navigable cells: ' ', '@', '$', '.', '*', '+'
// Excludes walls '#'
// Use on the shell (before placing elements) to get connectivity.
//

inline int count_free_cells(
    const std::vector<std::vector<char>>& board)
{
    int count = 0;

    for (const auto& row : board)
        for (char c : row)
            if (c != '#')
                count++;

    return count;
}

//
// RANDOM PLACEMENT
//

inline void placeRandom(
    std::vector<std::vector<char>>& board,
    char c)
{
    while (true)
    {
        int x =
            rand() % board.size();

        int y =
            rand() % board[0].size();

        if (board[x][y] == ' ')
        {
            board[x][y] = c;

            return;
        }
    }
}