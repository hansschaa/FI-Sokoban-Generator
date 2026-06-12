#pragma once

#include "../individual.h"
#include "../evaluator/evaluator.h"
#include "../mutations/move_mutation.h"
#include "../mutations/add_mutation.h"
#include "../mutations/remove_mutation.h"

#include <vector>

class EvolutionStrategy {

private:

    Evaluator evaluator;

    MoveMutation moveMutation;

    AddMutation addMutation;

    RemoveMutation removeMutation;

public:

    //
    // μ + λ PARAMETERS
    //

    int mu = 15;

    int lambda = 60;

    double mutationRate = 1.0;

    //
    // TERMINATION
    //

    int maxEvaluations = 2000;

    int stagnationLimit = 200;

    //
    // RUNTIME STATE
    //

    int evaluations = 0;

    //
    // MAIN ALGORITHM
    //

    Individual run(
        std::vector<Individual>& population);
};