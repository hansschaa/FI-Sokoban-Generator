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

            // Aquí el hijo hereda el genotipo Y el fitness ya calculado del padre
            Individual child =
                population[parentIndex];

            // BANDERA DE CONTROL: Asumimos que no necesita evaluación
            bool needsEvaluation = false;

            //
            // RANDOM MUTATION
            //

            double r_mut = (double)rand() / RAND_MAX;
            
            if (r_mut <= mutationRate) 
            {
                int mutationType = rand() % 3;
                bool success = false;

                if (mutationType == 0) {
                    success = moveMutation.apply(child);
                } else if (mutationType == 1) {
                    success = addMutation.apply(child);
                } else {
                    success = removeMutation.apply(child);
                }

                if (!success) {
                    // Si la mutación falló, descartamos a este hijo y pasamos al siguiente
                    continue;
                } else {
                    // La topología del nivel cambió. AHORA SÍ debemos evaluarlo.
                    needsEvaluation = true;
                }
            }
            // Si r_mut > mutationRate, needsEvaluation sigue siendo false.

            //
            // EVALUATION (El filtro anti-trampa)
            //

            if (needsEvaluation)
            {
                evaluator.evaluate(child);
                evaluations++;
            }

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

            // CORTE ESTRICTO DENTRO DEL BUCLE:
            // Si alcanzamos el límite exacto en medio de la generación de la población, abortamos.
            if (evaluations >= maxEvaluations)
            {
                break;
            }
        }

        //
        // STAGNATION UPDATE
        //

        if (offspring.empty())
        {
            std::cout
                << "WARNING: ALL "
                << lambda
                << " CHILDREN FAILED IN GEN "
                << generation
                << " (mutation invalid or NaN fitness)"
                << std::endl;
        }

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
        // Guard against combined being smaller than mu
        // (can happen if all offspring failed every generation)
        //

        population.clear();

        int toSelect =
            std::min(mu, (int)combined.size());

        for (int i = 0; i < toSelect; i++)
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