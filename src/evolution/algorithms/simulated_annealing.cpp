#include "../../../include/evolution/algorithms/simulated_annealing.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

double SimulatedAnnealing::acceptanceProbability(
    double currentScore,
    double newScore,
    double temperature)
{
    //
    // BETTER SOLUTION
    //

    if (newScore > currentScore)
    {
        return 1.0;
    }

    //
    // WORSE SOLUTION
    //

    return std::exp(
        (newScore - currentScore)
        / temperature);
}

Individual SimulatedAnnealing::run(
    Individual initial)
{
    //
    // INITIAL EVALUATION
    //

    evaluator.evaluate(initial);

    evaluations = 1;

    int stagnation = 0;

    //
    // CURRENT + BEST
    //

    Individual current =
        initial;

    Individual best =
        initial;

    //
    // INITIAL TEMP
    //

    temperature =
        initialTemperature;

    //
    // MAIN LOOP
    //

    while (
        evaluations < maxEvaluations &&
        temperature > 0.001 &&
        stagnation < stagnationLimit)
    {
        bool improved = false;

        //
        // CREATE NEIGHBOR
        //

        Individual neighbor =
            current;

        //
        // RANDOM MUTATION
        //

        int mutationType =
            rand() % 3;

        if (mutationType == 0)
        {
            moveMutation.apply(neighbor);
        }
        else if (mutationType == 1)
        {
            addMutation.apply(neighbor);
        }
        else
        {
            removeMutation.apply(neighbor);
        }

        //
        // EVALUATE
        //

        evaluator.evaluate(neighbor);

        evaluations++;

        //
        // ACCEPTANCE
        //

        double probability =
            acceptanceProbability(
                current.fitness,
                neighbor.fitness,
                temperature);

        double r =
            (double)rand() / RAND_MAX;

        if (probability > r)
        {
            current =
                neighbor;
        }

        //
        // UPDATE BEST
        //

        if (current.fitness > best.fitness)
        {
            best =
                current;

            improved = true;

            std::cout
                << "NEW BEST "
                << best.fitness
                << std::endl;
        }

        //
        // STAGNATION UPDATE
        //

        if (improved)
        {
            stagnation = 0;
        }
        else
        {
            stagnation++;
        }

        //
        // LOG
        //

        std::cout
            << "TEMP "
            << temperature
            << " | CURRENT "
            << current.fitness
            << " | BEST "
            << best.fitness
            << " | STAG "
            << stagnation
            << std::endl;

        //
        // COOLING
        //

        temperature *=
            (1.0 - coolingRate);
    }

    return best;
}