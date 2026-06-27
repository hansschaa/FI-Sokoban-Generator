#pragma once

#include "../individual.h"
#include "../mutations/mutation.h"
#include "../evaluator/evaluator.h"

#include "../mutations/move_mutation.h"
#include "../mutations/add_mutation.h"
#include "../mutations/remove_mutation.h"

class SimulatedAnnealing
{
public:

    //
    // PARAMETERS
    //

    double initialTemperature = 100.0;

    double coolingRate = 0.01;

    double temperature = 100.0;

    int maxEvaluations = 1000;

    int stagnationLimit = 200;

    int maxFailedAttempts = 100;

    MoveMutation moveMutation;

    AddMutation addMutation;

    RemoveMutation removeMutation;

    void setDeadlockMask(const std::vector<std::vector<bool>>& mask) {
        moveMutation.deadlock_mask = mask;
        addMutation.deadlock_mask = mask;
    }

    //
    // COMPONENTS
    //

    Evaluator evaluator;

    //
    // STATS
    //

    int evaluations = 0;

    //
    // METHODS
    //

    Individual run(
        Individual initial);

private:

    double acceptanceProbability(
        double currentScore,
        double newScore,
        double temperature);
};