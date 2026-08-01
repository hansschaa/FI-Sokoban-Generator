#pragma once

#include "../individual.h"
#include "../evaluator/evaluator.h"
#include "../mutations/move_mutation.h"
#include "../mutations/add_mutation.h"
#include "../mutations/remove_mutation.h"

#include <vector>
#include <chrono>
#include <functional>

class EvolutionStrategy {

private:

    MoveMutation moveMutation;

    AddMutation addMutation;

    RemoveMutation removeMutation;

public:

    Evaluator evaluator;
    bool use_parallel = true;
    bool adversarial_mode = false;

    void setDeadlockMask(const std::vector<std::vector<bool>>& mask) {
        moveMutation.deadlock_mask = mask;
        addMutation.deadlock_mask = mask;
    }

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
    // TIME LIMIT
    //

    std::chrono::time_point<std::chrono::high_resolution_clock> circuitStartTime;
    int maxCircuitTimeSeconds = -1; // -1 means no limit

    //
    // RUNTIME STATE
    //

    int evaluations = 0;

    // Optional callback triggered at the end of each generation (and for initial population).
    // Passes evaluation count, best fitness so far, and elapsed time in ms.
    std::function<void(int evals, double best_fitness, double time_ms, const Individual& best_ind)> on_progress;

    //
    // MAIN ALGORITHM
    //

    Individual run(
        std::vector<Individual>& population);
};