#pragma once

#include "../individual.h"
#include "fitness_type.h"

class Evaluator
{
public:

    FitnessType fitnessType =
        FitnessType::PUSHES;

    double evaluate(
        Individual& individual);

private:

    double computeEffectiveBranchingFactor(
        double expanded,
        double depth);
};