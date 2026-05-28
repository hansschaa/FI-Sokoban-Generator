#pragma once

#include <vector>
#include <string>

struct Pair {

    int i;
    int j;

    bool operator==(const Pair& other) const {
        return i == other.i && j == other.j;
    }
};

inline std::string board_to_string(
    const std::vector<std::vector<char>>& board)
{
    std::string result;

    for (auto& row : board) {

        for (char c : row)
            result += c;
    }

    return result;
}

inline int count_boxes(
    const std::vector<std::vector<char>>& board)
{
    int count = 0;

    for (auto& row : board)
        for (char c : row)
            if (c == '$' || c == '*')
                count++;

    return count;
}