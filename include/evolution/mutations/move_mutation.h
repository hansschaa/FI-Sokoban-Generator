#pragma once

#include "mutation.h"
#include "../utils/pair.h"

class MoveMutation : public Mutation
{
public:

    bool apply(Individual& ind) override;

private:

    bool moveCharacter(
        std::vector<std::vector<char>>& board,
        char target);

    std::vector<Pair> getPositions(
        const std::vector<std::vector<char>>& board,
        char c);

    Pair getRandomEmpty(
        const std::vector<std::vector<char>>& board);

    Pair getRandomDestination(
        const std::vector<std::vector<char>>& board,
        char target);
};