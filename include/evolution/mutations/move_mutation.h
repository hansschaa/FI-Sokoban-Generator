#pragma once

#include "mutation.h"
#include "../utils/pair.h"

class MoveMutation : public Mutation
{
public:

    void apply(Individual& ind) override;

private:

    void moveCharacter(
        std::vector<std::vector<char>>& board,
        char target);

    std::vector<Pair> getPositions(
        const std::vector<std::vector<char>>& board,
        char c);

    Pair getRandomEmpty(
        const std::vector<std::vector<char>>& board);
};