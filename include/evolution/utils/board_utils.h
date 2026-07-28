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

//
// COMPUTE DEADLOCK MASK
// Returns a boolean mask where true means placing a box there is a simple deadlock (corner or edge)
//
inline std::vector<std::vector<bool>> compute_deadlock_mask(
    const std::vector<std::vector<char>>& board)
{
    int r = board.size();
    int c = board.empty() ? 0 : board[0].size();
    
    std::vector<std::vector<bool>> mask(r, std::vector<bool>(c, false));
    
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if (board[i][j] == ' ') {
                bool top_wall = (i == 0 || board[i-1][j] == '#');
                bool bottom_wall = (i == r-1 || board[i+1][j] == '#');
                bool left_wall = (j == 0 || board[i][j-1] == '#');
                bool right_wall = (j == c-1 || board[i][j+1] == '#');
                
                bool vert_blocked = top_wall || bottom_wall;
                bool horiz_blocked = left_wall || right_wall;
                
                if (vert_blocked && horiz_blocked) {
                    mask[i][j] = true; // Corner
                } else if (vert_blocked) {
                    bool trapped = true;
                    // Scan left
                    for (int x = j - 1; x >= 0; x--) {
                        if (board[i][x] == '#') break;
                        // A goal along the corridor means a box could be placed there — not trapped
                        if (board[i][x] == '.' || board[i][x] == '*' || board[i][x] == '+') { trapped = false; break; }
                        if (!((i > 0 && board[i-1][x] == '#') || (i < r-1 && board[i+1][x] == '#'))) { trapped = false; break; }
                    }
                    // Scan right
                    for (int x = j + 1; x < c; x++) {
                        if (board[i][x] == '#') break;
                        if (board[i][x] == '.' || board[i][x] == '*' || board[i][x] == '+') { trapped = false; break; }
                        if (!((i > 0 && board[i-1][x] == '#') || (i < r-1 && board[i+1][x] == '#'))) { trapped = false; break; }
                    }
                    if (trapped) mask[i][j] = true;
                } else if (horiz_blocked) {
                    bool trapped = true;
                    // Scan up
                    for (int y = i - 1; y >= 0; y--) {
                        if (board[y][j] == '#') break;
                        if (board[y][j] == '.' || board[y][j] == '*' || board[y][j] == '+') { trapped = false; break; }
                        if (!((j > 0 && board[y][j-1] == '#') || (j < c-1 && board[y][j+1] == '#'))) { trapped = false; break; }
                    }
                    // Scan down
                    for (int y = i + 1; y < r; y++) {
                        if (board[y][j] == '#') break;
                        if (board[y][j] == '.' || board[y][j] == '*' || board[y][j] == '+') { trapped = false; break; }
                        if (!((j > 0 && board[y][j-1] == '#') || (j < c-1 && board[y][j+1] == '#'))) { trapped = false; break; }
                    }
                    if (trapped) mask[i][j] = true;
                }
            }
        }
    }
    
    return mask;
}

inline void placeRandom(std::vector<std::vector<char>>& board, char c, const std::vector<std::vector<bool>>& mask = {})
{
    // 1. Buscamos todos los espacios vacíos que existen en el tablero
    std::vector<std::pair<int, int>> vacios;
    for (size_t r = 0; r < board.size(); r++) {
        for (size_t c_col = 0; c_col < board[r].size(); c_col++) {
            if (board[r][c_col] == ' ') {
                if (c == '$' && !mask.empty() && r < mask.size() && c_col < mask[r].size() && mask[r][c_col]) {
                    continue; // Skip deadlock cell for boxes
                }
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