#pragma once

#include "../individual.h"
#include "../evaluator.h"
#include "../mutations/mutation.h"

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

    //
    // COMPONENTS
    //

    Mutation* mutation = nullptr;

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
        int currentScore,
        int newScore,
        double temperature);
};