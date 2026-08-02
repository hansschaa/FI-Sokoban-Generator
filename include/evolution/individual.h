#pragma once

#include <vector>
#include <string>

struct Individual {

    std::vector<std::vector<char>> board;

    double fitness = -1e9;

    int pushes = 0;
    int explored = 0;
    double hungarian_lb = 0;

    std::string parent_board_str;

    bool solved = false;
};