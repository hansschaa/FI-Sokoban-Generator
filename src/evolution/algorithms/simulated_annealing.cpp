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
    // SKIP RE-EVALUATION IF ALREADY EVALUATED
    // The individual comes pre-evaluated from main
    //

    if (initial.fitness == 0.0)
    {
        evaluator.evaluate(initial);
    }

    evaluations = 0;

    int stagnation = 0;

    int failedAttempts = 0;

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

        bool success = false;

        if (mutationType == 0)
        {
            success =
                moveMutation.apply(neighbor);
        }
        else if (mutationType == 1)
        {
            success =
                addMutation.apply(neighbor);
        }
        else
        {
            success =
                removeMutation.apply(neighbor);
        }

        //
        // INVALID MUTATION
        // Track consecutive failures to avoid infinite loop
        //

        if (!success)
        {
            failedAttempts++;

            if (failedAttempts >= maxFailedAttempts)
            {
                /*std::cout
                    << "TERMINATION: MAX FAILED ATTEMPTS"
                    << std::endl;*/

                break;
            }

            continue;
        }

        //
        // RESET FAILED ATTEMPTS ON SUCCESS
        //

        failedAttempts = 0;

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

            /*std::cout
                << "NEW BEST "
                << best.fitness
                << std::endl;*/
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

        /*std::cout
            << "TEMP "
            << temperature
            << " | CURRENT "
            << current.fitness
            << " | BEST "
            << best.fitness
            << " | STAG "
            << stagnation
            << std::endl;*/

        //
        // COOLING
        //

        temperature *=
            (1.0 - coolingRate);
    }

    if (evaluations >= maxEvaluations)
    {
        std::cout << "\n[SA] Criterio de Parada Alcanzado: MAX_EVALUATIONS (" << maxEvaluations << " evaluaciones)\n";
    }
    else if (stagnation >= stagnationLimit)
    {
        std::cout << "\n[SA] Criterio de Parada Alcanzado: STAGNATION (Sin mejoras por " << stagnationLimit << " generaciones)\n";
    }
    else if (temperature <= 0.001)
    {
        std::cout << "\n[SA] Criterio de Parada Alcanzado: MIN_TEMPERATURE (Temperatura llego a " << temperature << ")\n";
    }

    return best;
}