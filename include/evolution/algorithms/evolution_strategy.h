#pragma once

#include "../individual.h"
#include "../evaluator.h"
#include "../mutations/move_mutation.h"

class EvolutionStrategy {

private:

    Evaluator evaluator;

    MoveMutation mutation;

public:

    int maxEvaluations = 100;

    int stagnationLimit = 20;

    int evaluations = 0;

    Individual run(Individual current);
};