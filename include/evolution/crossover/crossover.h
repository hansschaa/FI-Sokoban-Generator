#pragma once

#include "../individual.h"

class Crossover
{
public:

    virtual Individual apply(
        const Individual& parent1,
        const Individual& parent2) = 0;

    virtual ~Crossover() = default;
};