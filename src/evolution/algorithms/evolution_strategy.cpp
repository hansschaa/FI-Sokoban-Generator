#include "../../../include/evolution/algorithms/evolution_strategy.h"

#include <iostream>
#include <cmath>

Individual EvolutionStrategy::run(
    Individual current)
{
    //
    // RESET STATE
    //

    evaluations = 0;

    int generation = 0;

    int stagnationCount = 0;

    //
    // INITIAL EVALUATION
    //

    std::cout << "EVALUATING INITIAL\n";

    evaluator.evaluate(current);

    evaluations++;

    std::cout << "INITIAL EVALUATED\n";

    //
    // GLOBAL BEST
    //

    Individual best = current;

    double bestFitness =
        best.fitness;

    //
    // MAIN LOOP
    //

    while (true) {

        std::cout
            << "GEN " << generation
            << " | BEST " << bestFitness
            << " | STAG " << stagnationCount
            << " | EVALS " << evaluations
            << "\n";

        //
        // CREATE OFFSPRING
        //

        Individual child = current;

        //
        // MUTATION
        //

        std::cout << "MUTATING\n";

        mutation.apply(child);

        //
        // EVALUATION
        //

        std::cout << "EVALUATING CHILD\n";

        evaluator.evaluate(child);

        evaluations++;

        std::cout << "CHILD EVALUATED\n";

        //
        // INVALID FITNESS CHECK
        //

        if (std::isnan(child.fitness))
        {
            std::cout
                << "INVALID FITNESS\n";

            continue;
        }

        //
        // REPLACEMENT
        // (1+1)-ES elitist replacement
        //

        if (child.fitness > current.fitness)
        {
            current = child;

            std::cout
                << "PARENT REPLACED\n";
        }

        //
        // GLOBAL BEST UPDATE
        //

        if (child.fitness > bestFitness)
        {
            best = child;

            bestFitness =
                child.fitness;

            stagnationCount = 0;

            std::cout
                << "NEW BEST FOUND\n";
        }
        else
        {
            stagnationCount++;
        }

        //
        // TERMINATION:
        // MAX EVALUATIONS
        //

        if (evaluations >= maxEvaluations)
        {
            std::cout
                << "Termination: MAX_EVALUATIONS"
                << std::endl;

            break;
        }

        //
        // TERMINATION:
        // STAGNATION
        //

        if (stagnationCount >= stagnationLimit)
        {
            std::cout
                << "Termination: STAGNATION"
                << std::endl;

            break;
        }

        generation++;
    }

    //
    // FINAL REPORT
    //

    std::cout << "\nFINAL BEST FITNESS: "
              << best.fitness
              << std::endl;

    return best;
}