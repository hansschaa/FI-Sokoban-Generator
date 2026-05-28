#pragma once

#include "crossover.h"

class BoardCrossover : public Crossover
{
public:

    Individual apply(
        const Individual& parent1,
        const Individual& parent2) override;
};