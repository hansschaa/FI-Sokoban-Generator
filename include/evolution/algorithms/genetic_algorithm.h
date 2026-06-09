#pragma once

#include <vector>

#include "../individual.h"

#include "../mutations/move_mutation.h"
#include "../mutations/add_mutation.h"
#include "../mutations/remove_mutation.h"

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

    int stagnationLimit = 20;

    int maxFailedAttempts = 10;
    
    double mutationRate = 1.0;

    //
    // STATE
    //

    int evaluations = 0;

    //
    // OPERATORS
    //

    Evaluator evaluator;

    MoveMutation moveMutation;

    AddMutation addMutation;

    RemoveMutation removeMutation;

    BoardCrossover crossover;

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