#include "../../../include/evolution/algorithms/evolution_strategy.h"
#include "../../../include/evolution/utils/board_utils.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>

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
    long long total_rejections = 0;
    //
    // RESET STATE
    //
    std::cerr << "[ES] Effective params: mu=" << mu 
              << ", lambda=" << lambda 
              << ", mutRate=" << mutationRate 
              << ", stagLimit=" << stagnationLimit << "\n";

    evaluations = 0;

    int generation = 0;

    int stagnationCount = 0;

    //
    // INITIAL EVALUATION
    //

    //std::cerr << "EVALUATING INITIAL POPULATION\n";

    unsigned int num_threads = use_parallel ? std::thread::hardware_concurrency() : 1;
    if (num_threads == 0) num_threads = 4;
    
    std::atomic<int> current_task{0};
    std::vector<std::future<void>> futures;

    auto eval_task = [&]() {
        while (true) {
            int i = current_task.fetch_add(1);
            if (i >= (int)population.size()) break;
            
            if (population[i].fitness <= -1e8 || population[i].fitness == 0) {
                Evaluator local_eval = evaluator;
                if (local_eval.heuristic_type == Heuristic::classifier_filter) {
                    local_eval.use_surrogate = false;
                }
                local_eval.evaluate(population[i]);
            }
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

    //std::cerr << "INITIAL POPULATION EVALUATED\n";

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
        on_progress(evaluations, best.fitness, elapsed_ms, best);
    }

    //
    // MAIN LOOP
    //

    int total_circuit_breakers = 0;
    int total_clone_fallbacks = 0;
    int total_clones_injected = 0;

    auto t_gen_wall_start = std::chrono::high_resolution_clock::now();
    double cumulative_mutation_ms = 0, cumulative_surrogate_ms = 0, cumulative_verification_ms = 0;
    int cumulative_verifications = 0;

    while (true)
    {
        bool improved = false;

        /*std::cerr
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

        auto t_mut_start = std::chrono::high_resolution_clock::now();
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
            child.parent_board_str = board_to_pretty_string(population[parentIndex].board);
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
                    if (success) child.needs_structural_audit = true;
                } else {
                    success = removeMutation.apply(child);
                    if (success) child.needs_structural_audit = true;
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

        auto t_mut_end = std::chrono::high_resolution_clock::now();
        double mutation_ms = std::chrono::duration<double, std::milli>(t_mut_end - t_mut_start).count();
        cumulative_mutation_ms += mutation_ms;

        // EVALUATION OF BATCH
        auto t_surr_start = std::chrono::high_resolution_clock::now();
        if (!batch_to_evaluate.empty()) {
            if (evaluator.use_surrogate && evaluator.heuristic_type != Heuristic::classifier_filter) {
                if (evaluator.heuristic_type != Heuristic::full_surrogate && evaluator.heuristic_type != Heuristic::hybrid_regressor) {
                    if (generation == 1 || generation == 10 || stagnationCount == stagnationLimit - 1) {
                        evaluator.evaluateDiagnostic(batch_to_evaluate, generation);
                    }
                }
                evaluator.evaluate_surrogate_batch(batch_to_evaluate);
            } else if (evaluator.heuristic_type == Heuristic::classifier_filter) {
                // 1. Pre-filtrado batch de deadlocks con el clasificador
                evaluator.filter_surrogate_batch(batch_to_evaluate);
                
                // 2. Evaluación en paralelo con solver A* real SOLO para los aprobados (que no se marcaron como -1e9)
                std::atomic<int> current_child{0};
                std::vector<std::future<void>> child_futures;
                for (unsigned int i = 0; i < num_threads; i++) {
                    child_futures.push_back(std::async(std::launch::async, [&]() {
                        while (true) {
                            int idx = current_child++;
                            if (idx >= (int)batch_to_evaluate.size()) break;
                            if (batch_to_evaluate[idx].fitness != -1e9 && batch_to_evaluate[idx].fitness != -2e9) {
                                auto orig_surr = evaluator.use_surrogate;
                                evaluator.use_surrogate = false;
                                double real_fit = evaluator.evaluate(batch_to_evaluate[idx]);
                                evaluator.use_surrogate = orig_surr;
                                if (real_fit == -1e9) {
                                    (*evaluator.classifier_false_positives)++;
                                }
                            }
                        }
                    }));
                }
                for (auto& f : child_futures) {
                    f.get();
                }
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
            total_rejections += (totalAttempts - generated);
        }
        auto t_surr_end = std::chrono::high_resolution_clock::now();
        double surrogate_ms = std::chrono::duration<double, std::milli>(t_surr_end - t_surr_start).count();
        cumulative_surrogate_ms += surrogate_ms;

        // PROCESS RESULTS
        for (auto& child : batch_to_evaluate) {
            if (!evaluator.use_surrogate || evaluator.heuristic_type == Heuristic::classifier_filter) {
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
            /*std::cerr
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

        // SELECT BEST μ
        // Guard against combined being smaller than mu
        int toSelect = std::min(mu, (int)combined.size());
        std::vector<Individual> next_population;
        
        int astar_failures = 0;
        const int MAX_FAILURES = 3;

        auto t_verify_start = std::chrono::high_resolution_clock::now();
        int gen_verifications = 0;
        for (int i = 0; i < (int)combined.size() && (int)next_population.size() < toSelect; i++)
        {
            auto& ind = combined[i];
            
            if (evaluator.use_surrogate && evaluator.heuristic_type == Heuristic::full_surrogate) {
                // Lógica de Auditoría por Riesgo Estructural para full_surrogate
                
                bool is_parent = false;
                for (const auto& parent : population) {
                    if (parent.board == ind.board) {
                        is_parent = true;
                        break;
                    }
                }
                
                // Si es un hijo estructuralmente riesgoso (Add/Remove) y el clasificador lo aprobó, auditamos.
                if (!is_parent && ind.needs_structural_audit && ind.fitness > 0) {
                    Evaluator astar_eval = evaluator;
                    astar_eval.use_surrogate = false;
                    astar_eval.heuristic_type = Heuristic::hungarian;
                    astar_eval.max_seconds = 5.0;
                    auto t_v0 = std::chrono::high_resolution_clock::now();
                    double true_fitness = astar_eval.evaluate(ind);
                    auto t_v1 = std::chrono::high_resolution_clock::now();
                    double v_ms = std::chrono::duration<double, std::milli>(t_v1 - t_v0).count();
                    gen_verifications++;
                    
                    if (true_fitness == -2e9) {
                        continue; // Timeout
                    }
                    
                    if (true_fitness > -1e8) {
                        // Confirmado como resolvible
                        ind.fitness = true_fitness;
                        ind.needs_structural_audit = false; // Validado permanentemente
                        next_population.push_back(ind);
                        if (true_fitness > best.fitness) {
                            best = ind;
                            improved = true;
                        }
                    } else {
                        // Falso positivo estructural (Deadlock descubierto)
                        total_circuit_breakers++; // Reusamos este contador para tracking
                        continue;
                    }
                } else {
                    // O es padre, o fue RandomWalk (confiamos en el fitness neuronal), o fue rechazado por el clasificador.
                    next_population.push_back(ind);
                }
            }
            else if (evaluator.use_surrogate && 
                evaluator.heuristic_type != Heuristic::classifier_filter && 
                evaluator.heuristic_type != Heuristic::full_surrogate && 
                evaluator.heuristic_type != Heuristic::hybrid_regressor) {
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
                    astar_eval.heuristic_type = Heuristic::hungarian;
                    astar_eval.max_seconds = 5.0;
                    auto t_v0 = std::chrono::high_resolution_clock::now();
                    double true_fitness = astar_eval.evaluate(ind);
                    auto t_v1 = std::chrono::high_resolution_clock::now();
                    double v_ms = std::chrono::duration<double, std::milli>(t_v1 - t_v0).count();
                    gen_verifications++;
                    std::cerr << "[PHASE_D] Verificacion A* #" << gen_verifications << " en Gen " << generation << ": " << v_ms << " ms (fitness=" << true_fitness << ")\n";
                    
                    if (true_fitness == -2e9) {
                        // TIMEOUT during verification!
                        // Inconclusive: Discard it, but DO NOT count as a circuit breaker failure
                        continue;
                    }
                    
                    if (true_fitness > -1e8) {
                        ind.fitness = true_fitness;
                        next_population.push_back(ind);
                        if (true_fitness > best.fitness) {
                            best = ind;
                            improved = true;
                        }
                    } else {
                        // A* says it's a deadlock (-1e9) but Surrogate accepted it!
                        if (adversarial_mode) {
                            // Adversarial Mining: Save the Hard Negative
                            std::filesystem::create_directories("hard_negatives");
                            std::string board_str;
                            for (const auto& row : ind.board) {
                                for (char c : row) board_str += c;
                                board_str += "\n";
                            }
                            size_t h = std::hash<std::string>{}(board_str);
                            std::string filepath = "hard_negatives/hard_negative_" + std::to_string(h) + ".sok";
                            std::ofstream out(filepath);
                            if (out.is_open()) {
                                out << "# surrogate_fitness: " << ind.fitness << "\n";
                                for (const auto& row : ind.board) {
                                    for (char c : row) out << c;
                                    out << "\n";
                                }
                            }
                            // Deceive the ES: keep the surrogate's fitness!
                            next_population.push_back(ind);
                        } else {
                            astar_failures++;
                            if (astar_failures == MAX_FAILURES) {
                                total_circuit_breakers++;
                            }
                            continue; // Discard False Positive
                        }
                    }
                } else {
                    next_population.push_back(ind);
                }
            } else {
                next_population.push_back(ind);
            }
        }
        
        auto t_verify_end = std::chrono::high_resolution_clock::now();
        double verify_ms = std::chrono::duration<double, std::milli>(t_verify_end - t_verify_start).count();
        cumulative_verification_ms += verify_ms;
        cumulative_verifications += gen_verifications;
        population = next_population;
        
        // If we rejected too many and couldn't fill mu, fill with clones of best
        if ((int)population.size() < toSelect) {
            total_clone_fallbacks++;
        }
        while ((int)population.size() < toSelect) {
            population.push_back(best);
            total_clones_injected++;
        }

        if (generation == 1 || generation == 2) {
            double mean = 0.0;
            for (const auto& ind : population) mean += ind.fitness;
            mean /= population.size();
            
            double var = 0.0;
            for (const auto& ind : population) var += (ind.fitness - mean) * (ind.fitness - mean);
            var /= population.size();
            
            std::cerr << "[DIVERSITY] Gen " << generation 
                      << " POP_STD: " << std::sqrt(var) 
                      << " (Mean: " << mean << ")" << std::endl;
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
        auto t_gen_now = std::chrono::high_resolution_clock::now();
        double gen_wall_ms = std::chrono::duration<double, std::milli>(t_gen_now - t_gen_wall_start).count();
        std::cerr << "[ES_TIMING] Gen " << generation 
                  << " | WallTime: " << (gen_wall_ms/1000.0) << "s"
                  << " | Evals: " << evaluations
                  << " | MutGen: " << cumulative_mutation_ms << "ms"
                  << " | Surrogate: " << cumulative_surrogate_ms << "ms"
                  << " | Verif(" << cumulative_verifications << "): " << cumulative_verification_ms << "ms"
                  << " | Best: " << best.fitness
                  << " | Stag: " << stagnationCount << "/" << stagnationLimit
                  << std::endl;

        if (on_progress) {
            Individual best_of_gen = population.empty() ? best : population[0];
            for (const auto& ind : population) {
                if (ind.fitness > best_of_gen.fitness) {
                    best_of_gen = ind;
                }
            }
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(now - circuitStartTime).count();
            on_progress(evaluations, best.fitness, elapsed_ms, best_of_gen);
        }
    }

    //
    // FINAL REPORT
    //
    
    std::cerr << "\n[ES STATS] Circuit Breaker (MAX_FAILURES) triggers: " << total_circuit_breakers << "\n";
    std::cerr << "[ES STATS] Surrogate Fallbacks: " << *(evaluator.surrogate_fallbacks) << "\n";
    std::cerr << "[ES STATS] Surrogate Regressor Calls: " << *(evaluator.surrogate_regressor_calls) << "\n";
    std::cerr << "[ES STATS] Classifier Deadlocks Filtered (Pre-A*): " << *(evaluator.classifier_deadlocks_filtered) << "\n";
    std::cerr << "[ES STATS] Classifier False Positives (Approved by Neural, Rejected by A*): " << *(evaluator.classifier_false_positives) << "\n";
    std::cerr << "[ES STATS] Hybrid Hungarian Delegations (box_count >= 6): " << *(evaluator.hybrid_hungarian_delegations) << "\n";
    std::cerr << "[ES STATS] Clone Fallback triggers: " << total_clone_fallbacks << " (Total Clones Injected: " << total_clones_injected << ")\n";
    std::cerr << "[ES STATS] Total Generations: " << generation << " | Total Evals: " << evaluations << "\n";

    /*std::cerr
        << "\nFINAL BEST FITNESS = "
        << best.fitness
        << std::endl;*/

    // Print circuit breaker stats to stderr for analysis
    std::cerr << "CB_REJECTIONS=" << total_rejections << std::endl;

    return best;
}