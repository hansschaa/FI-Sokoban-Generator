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

    if (on_progress) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - circuitStartTime).count();
        on_progress(evaluations, best.fitness, elapsed_ms);
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

        // EVALUATION OF BATCH
        if (!batch_to_evaluate.empty()) {
            if (evaluator.use_surrogate) {
                evaluator.evaluate_surrogate_batch(batch_to_evaluate);
            } else {
                std::atomic<int> current_child{0};
                std::vector<std::future<void>> child_futures;
                
                for (unsigned int i = 0; i < num_threads; i++) {
                    child_futures.push_back(std::async(std::launch::async, [&]() {
                        while (true) {
                            int idx = current_child++;
                            if (idx >= (int)batch_to_evaluate.size()) break;
                            evaluator.evaluate(batch_to_evaluate[idx]);
                        }
                    }));
                }
                for (auto& f : child_futures) {
                    f.get();
                }
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

        // SELECT BEST μ
        // Guard against combined being smaller than mu
        int toSelect = std::min(mu, (int)combined.size());
        std::vector<Individual> next_population;
        
        int astar_failures = 0;
        const int MAX_FAILURES = 3;

        for (int i = 0; i < (int)combined.size() && (int)next_population.size() < toSelect; i++)
        {
            auto& ind = combined[i];
            
            if (evaluator.use_surrogate) {
                // Check if this individual came from the old population (already verified)
                bool is_parent = false;
                for (const auto& parent : population) {
                    if (parent.board == ind.board) {
                        is_parent = true;
                        break;
                    }
                }
                
                if (!is_parent) {
                    if (astar_failures >= MAX_FAILURES) {
                        // Stop verifying to save time
                        continue;
                    }
                    
                    Evaluator astar_eval = evaluator;
                    astar_eval.use_surrogate = false;
                    double true_fitness = astar_eval.evaluate(ind);
                    
                    if (true_fitness > -1e8) {
                        ind.fitness = true_fitness;
                        next_population.push_back(ind);
                        if (true_fitness > best.fitness) {
                            best = ind;
                            improved = true;
                        }
                    } else {
                        astar_failures++;
                        continue; // Discard False Positive
                    }
                } else {
                    next_population.push_back(ind);
                }
            } else {
                next_population.push_back(ind);
            }
        }
        
        population = next_population;
        
        // If we rejected too many and couldn't fill mu, fill with clones of best
        while ((int)population.size() < toSelect) {
            population.push_back(best);
        }

        //
        // TERMINATION:
        // MAX EVALUATIONS
        //

        if (evaluations >= maxEvaluations)
        {
            std::cout
                << "\n[ES] Criterio de Parada Alcanzado: MAX_EVALUATIONS (" << maxEvaluations << " evaluaciones)\n";
            break;
        }

        //
        // TERMINATION:
        // STAGNATION
        //

        if (stagnationCount >= stagnationLimit)
        {
            std::cout
                << "\n[ES] Criterio de Parada Alcanzado: STAGNATION (Sin mejoras por " << stagnationLimit << " generaciones)\n";
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
                std::cout << "\n[ES] Criterio de Parada Alcanzado: TIME LIMIT (" << maxCircuitTimeSeconds << " segundos del circuito transcurridos)\n";
                break;
            }
        }

        generation++;
        std::cout << "[ES] Gen " << generation 
                  << " | Evals: " << evaluations << "/" << maxEvaluations 
                  << " | Stagnation: " << stagnationCount << "/" << stagnationLimit 
                  << " | Best Fit: " << best.fitness << std::endl;

        if (on_progress) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(now - circuitStartTime).count();
            on_progress(evaluations, best.fitness, elapsed_ms);
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