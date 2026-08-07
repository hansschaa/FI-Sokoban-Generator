#include "../../../include/evolution/algorithms/evolution_strategy.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <future>
#include <atomic>
#include <thread>

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

    //std::cout << "EVALUATING INITIAL POPULATION\n";

    unsigned int num_threads = use_parallel ? std::thread::hardware_concurrency() : 1;
    if (num_threads == 0) num_threads = 4;
    
    std::atomic<int> current_task{0};
    std::vector<std::future<void>> futures;

    auto eval_task = [&]() {
        while (true) {
            int i = current_task.fetch_add(1);
            if (i >= (int)population.size()) break;
            
            Evaluator local_eval = evaluator;
            local_eval.evaluate(population[i]);
        }
    };

    unsigned int threads_to_launch = std::min((unsigned int)population.size(), num_threads);
    for (unsigned int t = 0; t < threads_to_launch; t++) {
        futures.push_back(std::async(std::launch::async, eval_task));
    }
    for (auto& f : futures) {
        f.get();
    }

    evaluations += population.size();

    //std::cout << "INITIAL POPULATION EVALUATED\n";

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

    if (on_generation) {
        on_generation(generation, best);
    }

    //
    // MAIN LOOP
    //

    while (true)
    {
        bool improved = false;

        /*std::cout
            << "\nGEN " << generation
            << " | BEST " << best.fitness
            << " | STAG " << stagnationCount
            << " | EVALS " << evaluations
            << std::endl;*/

        //
        // OFFSPRING POPULATION
        //

        std::vector<Individual> offspring;

        //
        // GENERATE λ CHILDREN
        //

        int generated = 0;
        int totalAttempts = 0;
        const int maxAttempts = lambda * 10;

        std::vector<Individual> batch_to_evaluate;
        std::vector<Individual> batch_no_evaluation;

        while (
            generated < lambda &&
            (evaluations + batch_to_evaluate.size()) < (size_t)maxEvaluations &&
            totalAttempts < maxAttempts)
        {
            totalAttempts++;

            // RANDOM PARENT SELECTION
            int parentIndex = rand() % population.size();
            Individual child = population[parentIndex];
            bool needsEvaluation = false;

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

                if (!success) continue;
                needsEvaluation = true;
            }

            if (needsEvaluation) {
                batch_to_evaluate.push_back(child);
            } else {
                batch_no_evaluation.push_back(child);
            }
            generated++;
        }

        // PARALLEL EVALUATION OF BATCH
        if (!batch_to_evaluate.empty()) {
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
            
            evaluations += batch_to_evaluate.size();
        }

        // PROCESS RESULTS
        for (auto& child : batch_to_evaluate) {
            if (child.censored) {
                censored_evaluations++;
            }
            if (!std::isnan(child.fitness)) {
                offspring.push_back(child);
                if (child.fitness > best.fitness) {
                    best = child;
                    improved = true;
                }
            }
        }
        for (auto& child : batch_no_evaluation) {
            if (!std::isnan(child.fitness)) {
                offspring.push_back(child);
                if (child.fitness > best.fitness) {
                    best = child;
                    improved = true;
                }
            }
        }

        //
        // STAGNATION UPDATE
        //

        if (offspring.empty())
        {
            /*std::cout
                << "WARNING: ALL "
                << lambda
                << " CHILDREN FAILED IN GEN "
                << generation
                << " (mutation invalid or NaN fitness)"
                << std::endl;*/
        }

        if (improved)
        {
            stagnationCount = 0;
        }
        else
        {
            stagnationCount += batch_to_evaluate.size();
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
            std::cerr
                << "\n[ES] Criterio de Parada Alcanzado: MAX_EVALUATIONS (" << maxEvaluations << " evaluaciones)\n";
            break;
        }

        //
        // TERMINATION:
        // STAGNATION
        //

        if (stagnationCount >= stagnationLimit)
        {
            std::cerr
                << "\n[ES] Criterio de Parada Alcanzado: STAGNATION (Sin mejoras por " << stagnationLimit << " evaluaciones)\n";
            break;
        }

        //
        // TERMINATION:
        // TIME LIMIT
        //

        if (maxCircuitTimeSeconds > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - circuitStartTime).count();
            if (duration >= maxCircuitTimeSeconds) {
                std::cerr << "\n[ES] Criterio de Parada Alcanzado: TIME LIMIT (" << maxCircuitTimeSeconds << " segundos del circuito transcurridos)\n";
                break;
            }
        }

        generation++;
        std::cout << "[ES] Gen " << generation 
                  << " | Evals: " << evaluations << "/" << maxEvaluations 
                  << " | Stagnation: " << stagnationCount << "/" << stagnationLimit 
                  << " | Best Fit: " << best.fitness << std::endl;

        if (on_generation) {
            on_generation(generation, best);
        }
    }

    //
    // FINAL REPORT
    //

    /*std::cout
        << "\nFINAL BEST FITNESS = "
        << best.fitness
        << std::endl;*/

    return best;
}