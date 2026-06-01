#pragma once

#include "../individual.h"

class Crossover
{
public:

    virtual bool apply(
        const Individual& parent1,
        const Individual& parent2,
        Individual& child) = 0;

    virtual ~Crossover() = default;
};