#include "../../../include/evolution/algorithms/genetic_algorithm.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <future>
#include <atomic>
#include <thread>

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
    // (Omitida: La población inicial ya viene evaluada con A* desde main_evolution.cpp)
    //
    
    unsigned int num_threads = use_parallel ? std::thread::hardware_concurrency() : 1;
    if (num_threads == 0) num_threads = 4;
    evaluations += population.size();

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

        std::vector<Individual> batch_to_evaluate;

        while (
            generated < offspringSize &&
            (evaluations + batch_to_evaluate.size()) < (size_t)maxEvaluations &&
            totalAttempts < maxAttempts)
        {
            totalAttempts++;

            // SELECTION
            Individual p1 = tournamentSelection(population);
            Individual p2 = tournamentSelection(population);

            // CROSSOVER
            Individual child;
            bool valid = false;

            double r_cross = (double)rand() / RAND_MAX;
            if (r_cross <= crossoverRate)
            {
                valid = crossover.apply(p1, p2, child);
                if (!valid) continue;
            }
            else
            {
                child = p1;
                valid = true;
            }

            // MUTATION
            double r_mut = (double)rand() / RAND_MAX;
            if (r_mut <= mutationRate) 
            {
                bool success = applyRandomMutation(child);
                if (!success) continue;
            }

            batch_to_evaluate.push_back(child);
            generated++;
        }

        // PARALLEL EVALUATION OF BATCH
        if (!batch_to_evaluate.empty()) {
            if (evaluator.use_surrogate) {
                evaluator.evaluate_surrogate_batch(batch_to_evaluate);
            } else {
                std::atomic<int> current_child{0};
                std::vector<std::future<void>> child_futures;
                
                auto child_eval_task = [&]() {
                    while(true) {
                        int i = current_child.fetch_add(1);
                        if (i >= (int)batch_to_evaluate.size()) break;
                        
                        Evaluator local_eval = evaluator;
                        local_eval.evaluate(batch_to_evaluate[i]);
                    }
                };
                
                unsigned int c_threads = std::min((unsigned int)batch_to_evaluate.size(), num_threads);
                for (unsigned int t = 0; t < c_threads; t++) {
                    child_futures.push_back(std::async(std::launch::async, child_eval_task));
                }
                for(auto& f : child_futures) f.get();
            }
            
            evaluations += batch_to_evaluate.size();
        }

        // PROCESS RESULTS
        for (auto& child : batch_to_evaluate) {
            if (!evaluator.use_surrogate) {
                if (child.fitness > best.fitness) {
                    best = child;
                    improved = true;
                }
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
        // ELITIST REPLACEMENT WITH SURROGATE VERIFICATION
        //

        population.clear();

        //
        // KEEP GLOBAL BEST (Already verified)
        //

        population.push_back(best);

        //
        // ADD BEST OFFSPRING (Verify if using surrogate)
        //
        int astar_failures = 0;
        const int MAX_FAILURES = 3;

        for (int i = 0; i < (int)offspring.size() && (int)population.size() < populationSize; i++)
        {
            if (evaluator.use_surrogate) {
                if (astar_failures >= MAX_FAILURES) {
                    // Surrogate is hallucinating false positives too much this generation.
                    // Stop verifying to save time, rest of population will be filled with clones of best.
                    break;
                }
                
                Evaluator astar_eval = evaluator;
                astar_eval.use_surrogate = false;
                double true_fitness = astar_eval.evaluate(offspring[i]);
                
                if (true_fitness > -1e8) {
                    offspring[i].fitness = true_fitness;
                    population.push_back(offspring[i]);
                    if (true_fitness > best.fitness) {
                        best = offspring[i];
                        improved = true;
                    }
                } else {
                    astar_failures++;
                }
            } else {
                population.push_back(offspring[i]);
            }
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

        if (maxCircuitTimeSeconds > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - circuitStartTime).count();
            if (duration >= maxCircuitTimeSeconds) {
                std::cout << "\n[GA] Criterio de Parada Alcanzado: TIME LIMIT (" << maxCircuitTimeSeconds << " segundos del circuito transcurridos)\n";
                break;
            }
        }
    }

    //
    // TERMINATION REPORT
    //

    if (evaluations >= maxEvaluations)
    {
        std::cout
            << "\n[GA] Criterio de Parada Alcanzado: MAX_EVALUATIONS (" << maxEvaluations << " evaluaciones)\n";
    }

    if (stagnation >= stagnationLimit)
    {
        std::cout
            << "\n[GA] Criterio de Parada Alcanzado: STAGNATION (Sin mejoras por " << stagnationLimit << " generaciones)\n";
    }

    return best;
}