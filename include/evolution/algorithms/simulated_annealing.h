#pragma once

#include "../individual.h"
#include <chrono>
#include <functional>
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

    //
    // TIME LIMIT
    //

    std::chrono::time_point<std::chrono::high_resolution_clock> circuitStartTime;
    int maxCircuitTimeSeconds = -1; // -1 means no limit

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
    
    // Optional callback triggered at the end of each generation (and for initial population).
    // Passes evaluation count, best fitness so far, and elapsed time in ms.
    std::function<void(int evals, double best_fitness, double time_ms)> on_progress;

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