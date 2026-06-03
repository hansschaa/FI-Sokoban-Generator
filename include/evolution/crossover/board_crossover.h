#pragma once

#include "crossover.h"
#include "../utils/pair.h"

class BoardCrossover : public Crossover
{
public:

    //
    // CROSSOVER SPACING
    // Length of the interesting region to copy
    // Default: 2
    //

    int crossoverSpacing = 2;

    bool apply(
        const Individual& parent1,
        const Individual& parent2,
        Individual& child) override;

private:

    struct CrossPair
    {
        Pair pivot;
        Pair direction;
    };

    //
    // GET INTERESTING REGIONS
    //

    std::vector<CrossPair> getInterestingRegions(
        const std::vector<std::vector<char>>& board);

    //
    // APPLY REGION TO CHILD
    //

    void applyRegion(
        std::vector<std::vector<char>>& board,
        const std::vector<std::vector<char>>& source,
        const CrossPair& region);

    //
    // COUNT ELEMENTS
    //

    int countBoxes(
        const std::vector<std::vector<char>>& board);

    int countGoals(
        const std::vector<std::vector<char>>& board);

    int countPlayers(
        const std::vector<std::vector<char>>& board);

    //
    // GET RANDOM EMPTY CELL
    //

    Pair getRandomEmpty(
        const std::vector<std::vector<char>>& board);

    //
    // REPAIR ILLEGAL BOARD
    //

    void repairIllegal(
        std::vector<std::vector<char>>& board);

    //
    // VALIDATE STRUCTURE
    // Does not call solver
    //

    bool isStructurallyValid(
        const std::vector<std::vector<char>>& board);
};