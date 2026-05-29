#include "../../../include/evolution/algorithms/genetic_algorithm.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>

//
// TOURNAMENT SELECTION
//

Individual GeneticAlgorithm::tournamentSelection(
    const std::vector<Individual>& population)
{
    int a =
        rand() % population.size();

    int b =
        rand() % population.size();

    if (population[a].fitness >
        population[b].fitness)
    {
        return population[a];
    }

    return population[b];
}

//
// RANDOM MUTATION
//

Individual GeneticAlgorithm::applyRandomMutation(
    Individual child)
{
    int r =
        rand() % 3;

    if (r == 0)
    {
        moveMutation.apply(child);
    }
    else if (r == 1)
    {
        addMutation.apply(child);
    }
    else
    {
        removeMutation.apply(child);
    }

    return child;
}

//
// MAIN GA
//

Individual GeneticAlgorithm::run(
    std::vector<Individual>& population)
{
    evaluations = 0;

    int stagnation = 0;

    //
    // INITIAL EVALUATION
    //

    for (auto& ind : population)
    {
        evaluator.evaluate(ind);

        evaluations++;
    }

    //
    // INITIAL BEST
    //

    Individual best =
        population[0];

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
            << "EVALS "
            << evaluations
            << " | BEST "
            << best.fitness
            << " | STAG "
            << stagnation
            << std::endl;

        //
        // NEW POPULATION
        //

        std::vector<Individual> offspring;

        //
        // GENERATE CHILDREN
        //

        for (int i = 0;
             i < offspringSize;
             i++)
        {
            //
            // SELECTION
            //

            Individual p1 =
                tournamentSelection(population);

            Individual p2 =
                tournamentSelection(population);

            //
            // CROSSOVER
            //

            Individual child =
                crossover.apply(p1, p2);

            //
            // MUTATION
            //

            child =
                applyRandomMutation(child);

            //
            // EVALUATION
            //

            evaluator.evaluate(child);

            evaluations++;

            //
            // UPDATE BEST
            //

            if (child.fitness > best.fitness)
            {
                best = child;

                improved = true;

                std::cout
                    << "NEW BEST "
                    << best.fitness
                    << std::endl;
            }

            offspring.push_back(child);
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
        // SORT OFFSPRING
        //

        std::sort(
            offspring.begin(),
            offspring.end(),
            [](const Individual& a,
               const Individual& b)
        {
            return a.fitness > b.fitness;
        });

        //
        // ELITIST GENERATIONAL REPLACEMENT
        //

        population.clear();

        //
        // KEEP GLOBAL BEST
        //

        population.push_back(best);

        //
        // FILL REST
        //

        for (int i = 0;
             i < offspringSize - 1;
             i++)
        {
            population.push_back(offspring[i]);
        }

        //
        // TERMINATION
        //

        if (evaluations >= maxEvaluations)
        {
            std::cout
                << "TERMINATION: MAX EVALUATIONS"
                << std::endl;

            break;
        }

        if (stagnation >= stagnationLimit)
        {
            std::cout
                << "TERMINATION: STAGNATION"
                << std::endl;

            break;
        }
    }

    return best;
}