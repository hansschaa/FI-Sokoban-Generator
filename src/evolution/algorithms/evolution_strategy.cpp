#include "../../../include/evolution/algorithms/evolution_strategy.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

Individual EvolutionStrategy::run(
    std::vector<Individual>& population)
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

    std::cout << "EVALUATING INITIAL POPULATION\n";

    for (auto& ind : population)
    {
        evaluator.evaluate(ind);

        evaluations++;
    }

    std::cout << "INITIAL POPULATION EVALUATED\n";

    //
    // FIND INITIAL BEST
    //

    Individual best = population[0];

    for (auto& ind : population)
    {
        if (ind.fitness > best.fitness)
        {
            best = ind;
        }
    }

    //
    // MAIN LOOP
    //

    while (true)
    {
        bool improved = false;

        std::cout
            << "\nGEN " << generation
            << " | BEST " << best.fitness
            << " | STAG " << stagnationCount
            << " | EVALS " << evaluations
            << std::endl;

        //
        // OFFSPRING POPULATION
        //

        std::vector<Individual> offspring;

        //
        // GENERATE λ CHILDREN
        //

        for (int i = 0; i < lambda; i++)
        {
            //
            // RANDOM PARENT SELECTION
            //

            int parentIndex =
                rand() % population.size();

            Individual child =
                population[parentIndex];

            //
            // RANDOM MUTATION
            //

            int mutationType =
                rand() % 3;

            if (mutationType == 0)
            {
                moveMutation.apply(child);

                std::cout
                    << "MOVE MUTATION\n";
            }
            else if (mutationType == 1)
            {
                addMutation.apply(child);

                std::cout
                    << "ADD MUTATION\n";
            }
            else
            {
                removeMutation.apply(child);

                std::cout
                    << "REMOVE MUTATION\n";
            }

            //
            // EVALUATION
            //

            evaluator.evaluate(child);

            evaluations++;

            //
            // INVALID FITNESS
            //

            if (std::isnan(child.fitness))
            {
                std::cout
                    << "INVALID FITNESS\n";

                continue;
            }

            //
            // SAVE CHILD
            //

            offspring.push_back(child);

            //
            // GLOBAL BEST UPDATE
            //

            if (child.fitness > best.fitness)
            {
                best = child;

                improved = true;

                std::cout
                    << "NEW BEST = "
                    << best.fitness
                    << std::endl;
            }
        }

        //
        // STAGNATION UPDATE
        //

        if (improved)
        {
            stagnationCount = 0;
        }
        else
        {
            stagnationCount++;
        }

        //
        // COMBINE POPULATIONS
        // (μ + λ)-ES
        //

        std::vector<Individual> combined =
            population;

        combined.insert(
            combined.end(),
            offspring.begin(),
            offspring.end());

        //
        // SORT BY FITNESS DESC
        //

        std::sort(
            combined.begin(),
            combined.end(),
            [](const Individual& a,
               const Individual& b)
        {
            return a.fitness > b.fitness;
        });

        //
        // SELECT BEST μ
        //

        population.clear();

        for (int i = 0; i < mu; i++)
        {
            population.push_back(combined[i]);
        }

        //
        // TERMINATION:
        // MAX EVALUATIONS
        //

        if (evaluations >= maxEvaluations)
        {
            std::cout
                << "\nTermination: MAX_EVALUATIONS"
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
                << "\nTermination: STAGNATION"
                << std::endl;

            break;
        }

        generation++;
    }

    //
    // FINAL REPORT
    //

    std::cout
        << "\nFINAL BEST FITNESS = "
        << best.fitness
        << std::endl;

    return best;
}