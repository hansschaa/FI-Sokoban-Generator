#pragma once

#include <vector>
#include <chrono>

#include "../individual.h"

#include "../mutations/move_mutation.h"
#include "../mutations/add_mutation.h"
#include "../mutations/remove_mutation.h"
#include <functional>

#include "../crossover/board_crossover.h"
#include "../evaluator/evaluator.h"

class GeneticAlgorithm
{
public:

    //
    // PARAMETERS
    //

    int offspringSize = 20;

    int maxEvaluations = 500;

    int stagnationLimit = 50;

    //
    // TIME LIMIT
    //

    std::chrono::time_point<std::chrono::high_resolution_clock> circuitStartTime;
    int maxCircuitTimeSeconds = -1; // -1 means no limit

    int maxFailedAttempts = 10;
    
    double mutationRate = 1.0;
    double crossoverRate = 1.0; // Valor por defecto

    //
    // STATE
    //

    int evaluations = 0;
    
    // Optional callback triggered at the end of each generation (and for initial population).
    // Passes evaluation count, best fitness so far, and elapsed time in ms.
    std::function<void(int evals, double best_fitness, double time_ms)> on_progress;

    //
    // OPERATORS
    //

    Evaluator evaluator;

    MoveMutation moveMutation;

    AddMutation addMutation;

    RemoveMutation removeMutation;

    bool use_parallel = true;

    std::vector<std::vector<bool>> deadlock_mask;

    BoardCrossover crossover;

    void setDeadlockMask(const std::vector<std::vector<bool>>& mask) {
        moveMutation.deadlock_mask = mask;
        addMutation.deadlock_mask = mask;
    }

    //
    // METHODS
    //

    Individual run(
        std::vector<Individual>& population);

    Individual tournamentSelection(
        const std::vector<Individual>& population);

    bool applyRandomMutation(
        Individual& child); 
};