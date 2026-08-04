#pragma once

#include <vector>

struct Individual {

    std::vector<std::vector<char>> board;

    double fitness = -1e9;

    int pushes = 0;
    int explored = 0;

    bool solved = false;
    bool censored = false;
};