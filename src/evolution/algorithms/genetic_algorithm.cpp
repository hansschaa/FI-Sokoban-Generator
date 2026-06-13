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

bool GeneticAlgorithm::applyRandomMutation(
    Individual& child)
{
    int r =
        rand() % 3;

    if (r == 0)
    {
        return
            moveMutation.apply(child);
    }
    else if (r == 1)
    {
        return
            addMutation.apply(child);
    }

    return
        removeMutation.apply(child);
}

//
// MAIN GA
//

Individual GeneticAlgorithm::run(
    std::vector<Individual>& population)
{
    evaluations = 0;

    const int populationSize =
        population.size();

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

    while (
        evaluations < maxEvaluations &&
        stagnation < stagnationLimit)
    {
        bool improved = false;

        /*std::cout
            << "EVALS "
            << evaluations
            << " | BEST "
            << best.fitness
            << " | STAG "
            << stagnation
            << std::endl;*/

        //
        // OFFSPRING
        //

        std::vector<Individual> offspring;

        //
        // GENERATE CHILDREN
        // Loop por intentos totales, no por hijos generados,
        // para que crossover/mutacion fallidos no consuman slots
        //

        int generated    = 0;
        int totalAttempts = 0;
        const int maxAttempts = offspringSize * 10;

        while (
            generated < offspringSize &&
            evaluations < maxEvaluations &&
            totalAttempts < maxAttempts)
        {
            totalAttempts++;

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

            Individual child;
            bool valid = false;

            // Tirar los dados para el cruce
            double r_cross = (double)rand() / RAND_MAX;

            if (r_cross <= crossoverRate)
            {
                // Ocurre el cruce
                valid = crossover.apply(p1, p2, child);
                
                if (!valid)
                {
                    continue; // Si falla el cruce, descartamos el intento
                }
            }
            else
            {
                // No hay cruce: el hijo hereda directamente del padre 1
                child = p1;
                valid = true;
            }

            //
            // MUTATION
            //

            // AÑADIDO: Chequeo de probabilidad de mutación
            double r_mut = (double)rand() / RAND_MAX;
            
            if (r_mut <= mutationRate) 
            {
                bool success = applyRandomMutation(child);

                if (!success)
                {
                    continue; // Si decide mutar y falla, descartamos el intento
                }
            }
            // Si r_mut > mutationRate, el child simplemente no muta y pasa tal cual (producto del crossover)

            //
            // EVALUATION
            //

            evaluator.evaluate(child);

            evaluations++;

            generated++;

            //
            // UPDATE BEST
            //

            if (child.fitness > best.fitness)
            {
                best = child;

                improved = true;

                /*std::cout
                    << "NEW BEST "
                    << best.fitness
                    << std::endl;*/
            }

            offspring.push_back(child);
        }

        if (totalAttempts >= maxAttempts)
        {
            /*std::cout
                << "WARNING: OFFSPRING GENERATION HIT ATTEMPT LIMIT ("
                << maxAttempts
                << "), GENERATED "
                << generated
                << "/"
                << offspringSize
                << std::endl;*/
        }

        //
        // STAGNATION
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
        // NO VALID OFFSPRING
        //

        if (offspring.empty())
        {
            /*std::cout
                << "WARNING: NO VALID OFFSPRING"
                << std::endl;*/

            continue;
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
                return a.fitness >
                       b.fitness;
            });

        //
        // ELITIST REPLACEMENT
        //

        population.clear();

        //
        // KEEP GLOBAL BEST
        //

        population.push_back(best);

        //
        // ADD BEST OFFSPRING
        //

        for (int i = 0;
             i < (int)offspring.size() &&
             (int)population.size() < populationSize;
             i++)
        {
            population.push_back(
                offspring[i]);
        }

        //
        // RECOVER POPULATION SIZE
        //

        {
            int recoveryFailed = 0;

            while (
                (int)population.size() <
                populationSize &&
                evaluations < maxEvaluations)
            {
                Individual clone =
                    best;

                bool success =
                    applyRandomMutation(clone);

                if (!success)
                {
                    recoveryFailed++;

                    if (recoveryFailed >= maxFailedAttempts)
                    {
                        /*std::cout
                            << "WARNING: RECOVERY GAVE UP AFTER "
                            << maxFailedAttempts
                            << " FAILED MUTATIONS"
                            << std::endl;*/

                        break;
                    }

                    continue;
                }

                recoveryFailed = 0;

                evaluator.evaluate(clone);

                evaluations++;

                population.push_back(clone);
            }
        }

        //
        // SAFETY
        //

        if (population.empty())
        {
            std::cerr
                << "ERROR: population collapsed"
                << std::endl;

            break;
        }
    }

    //
    // TERMINATION REPORT
    //

    /*if (evaluations >= maxEvaluations)
    {
        std::cout
            << "TERMINATION: MAX EVALUATIONS"
            << std::endl;
    }

    if (stagnation >= stagnationLimit)
    {
        std::cout
            << "TERMINATION: STAGNATION"
            << std::endl;
    }*/

    return best;
}