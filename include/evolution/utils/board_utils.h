#pragma once

#include <vector>
#include <string>
#include <cstdlib>
#include <stdexcept> 
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

inline void placeRandom(std::vector<std::vector<char>>& board, char c)
{
    // 1. Buscamos todos los espacios vacíos que existen en el tablero
    std::vector<std::pair<int, int>> vacios;
    for (size_t r = 0; r < board.size(); r++) {
        for (size_t c_col = 0; c_col < board[r].size(); c_col++) {
            if (board[r][c_col] == ' ') {
                vacios.push_back({(int)r, (int)c_col});
            }
        }
    }

    // 2. Si no hay espacios vacíos, lanzamos la excepción que atrapará main_irace.cpp
    if (vacios.empty()) {
        throw std::runtime_error(std::string("TABLERO LLENO al intentar poner: ") + c);
    }

    // 3. Elegimos uno al azar de los disponibles
    int idx = rand() % vacios.size();
    board[vacios[idx].first][vacios[idx].second] = c;
}