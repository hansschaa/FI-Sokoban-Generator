#include "../../../include/evolution/evaluator/evaluator.h"
#include "../../../include/game_solver.h"
#include "../../../include/evolution/utils/board_utils.h"
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include "../../../include/httplib.h"
#include "../../../include/nlohmann/json.hpp"

using json = nlohmann::json;


void Evaluator::registrar_tablero_critico(const std::vector<std::vector<char>>& board) {
}

double Evaluator::evaluate(Individual& individual)
{
    if (this->use_surrogate) {
        std::vector<Individual> pop = { individual };
        // Disable use_surrogate temporarily to avoid infinite recursion if fallback happens
        this->use_surrogate = false; 
        evaluate_surrogate_batch(pop);
        this->use_surrogate = true;
        individual.fitness = pop[0].fitness;
        return individual.fitness;
    }

    std::string level = board_to_string(individual.board);
    unsigned int rows = individual.board.size();
    unsigned int cols = individual.board[0].size();

    game_solver solver(level, rows, cols, 64);
    std::vector<game_node> solution;

    bool needs_path_simulator = (fitnessType == FitnessType::FO3_SOL_EFF_BF);

    registrar_tablero_critico(individual.board);

    solver.enable_advanced_deadlocks = true;

    // We ALWAYS use Hungarian for true ground-truth evaluation, bypassing LibTorch completely.
    // The neural heuristic is exclusively used via the Python Surrogate Server in evaluate_surrogate_batch.
    auto stats = solver.test_template(Method::a_star, Heuristic::hungarian, solution, needs_path_simulator, nullptr, this->max_seconds);
    
    individual.hungarian_lb = stats.initial_optimal_distance;

    if (stats.status == SolveStatus::TIMEOUT)
    {
        individual.fitness = -2e9;
        return individual.fitness;
    }

    if (stats.status != SolveStatus::SOLVED || stats.pushes <= 1)
    {
        individual.fitness = -1e9;
        return individual.fitness;
    }

    // EXTRAER EL FITNESS CORRECTO
    switch (fitnessType)
    {
        case FitnessType::FO1_PUSHES:
        case FitnessType::FO6_PUSHES_AND_SPEED:
            individual.fitness = stats.pushes;
            break;
            
        case FitnessType::FO2_ASTAR_EFF_BF:
            // Para MINIMIZAR el Branching Effectivo usando metaheurísticas 
            // que MAXIMIZAN, invertimos el signo.
            // (Ej: -1.2 es "mejor" que -2.5, guiando la evolución hacia la baja).
            individual.fitness = -stats.branching_effective;
            break;
            
        case FitnessType::FO3_SOL_EFF_BF:
            if (stats.path_stats_calculated) {
                individual.fitness = stats.path_stats.get_branching_effective_avg();
            } else {
                individual.fitness = 0.0; 
            }
            break;
            
        case FitnessType::FO4_DEADLOCKS:
            individual.fitness = stats.deadlocks;
            break;

        case FitnessType::FO5_REPEATED_NODES:
            individual.fitness = stats.repeated_nodes;
            break;
    }

    return individual.fitness;
}

// Puedes borrar computeEffectiveBranchingFactor de este archivo ya que ahora
// usaremos las métricas precisas reales que calcula game_solver y path_simulator.

void Evaluator::evaluate_surrogate_batch(std::vector<Individual>& population)
{
    if (population.empty()) return;

    std::vector<size_t> surrogate_indices;
    std::vector<size_t> hungarian_indices;

    for (size_t i = 0; i < population.size(); ++i) {
        // Hybrid Switch: Usar Hungarian puro para tableros con 6+ cajas
        if (count_boxes(population[i].board) >= 6) {
            hungarian_indices.push_back(i);
        } else {
            surrogate_indices.push_back(i);
        }
    }

    // Delegación directa a Hungarian puro para box_count >= 6
    if (!hungarian_indices.empty()) {
        (*this->hybrid_hungarian_delegations) += hungarian_indices.size();
        auto original_heuristic = this->heuristic_type;
        auto original_max_sec = this->max_seconds;
        bool original_surrogate = this->use_surrogate;

        this->heuristic_type = Heuristic::hungarian;
        this->use_surrogate = false;
        this->max_seconds = 5.0; // Verificación rápida en Hungarian

        for (size_t idx : hungarian_indices) {
            evaluate(population[idx]);
        }

        this->heuristic_type = original_heuristic;
        this->use_surrogate = original_surrogate;
        this->max_seconds = original_max_sec;
    }

    if (surrogate_indices.empty()) return;

    if (this->heuristic_type == Heuristic::hybrid_regressor) {
        std::vector<size_t> solvable_for_regressor;
        
        auto original_heuristic = this->heuristic_type;
        auto original_max_sec = this->max_seconds;
        bool original_surrogate = this->use_surrogate;
        
        this->heuristic_type = Heuristic::hungarian;
        this->use_surrogate = false;
        this->max_seconds = 5.0; // User specified 5.0s timeout

        for (size_t idx : surrogate_indices) {
            auto& ind = population[idx];
            
            // Fast deadlock check
            std::string flat_str = board_to_string(ind.board);
            unsigned int rows = ind.board.size();
            unsigned int cols = ind.board.empty() ? 0 : ind.board[0].size();
            game_solver fast_solver(flat_str, rows, cols, 16);
            fast_solver.enable_advanced_deadlocks = true;
            
            bool simple_deadlock = false;
            for (size_t i = 0; i < ind.board.size(); i++) {
                for (size_t j = 0; j < ind.board[i].size(); j++) {
                    if (ind.board[i][j] == '$' || ind.board[i][j] == '*') {
                        point p(i, j);
                        if (fast_solver.lk.is_locked(p, ind.board) || fast_solver.lk.is_freeze_deadlock(p, ind.board)) {
                            simple_deadlock = true;
                            break;
                        }
                    }
                }
                if (simple_deadlock) break;
            }
            
            if (simple_deadlock) {
                ind.fitness = -1e9;
                continue;
            }
            
            auto t_del_start = std::chrono::high_resolution_clock::now();
            evaluate(population[idx]);
            auto t_del_end = std::chrono::high_resolution_clock::now();
            std::cerr << "[TIMING_PHASE] (b.4) Delegacion hibrida legal (A*): " << std::chrono::duration<double, std::milli>(t_del_end - t_del_start).count() << " ms\n";
            
            // If it timed out, evaluate() sets fitness very low or negative, treat it as discarded
            // If it's solvable, fitness > 0
            if (ind.fitness > 0) {
                solvable_for_regressor.push_back(idx);
            } else {
                ind.fitness = -1e9; // Ensure discarded
            }
        }
        
        this->heuristic_type = original_heuristic;
        this->use_surrogate = original_surrogate;
        this->max_seconds = original_max_sec;
        
        if (solvable_for_regressor.empty()) return;
        
        // Prepare payload for solvable ones
        json payload;
        payload["boards"] = json::array();
        for (size_t idx : solvable_for_regressor) {
            const auto& ind = population[idx];
            json item;
            item["board"] = board_to_pretty_string(ind.board);
            item["parent_board"] = ind.parent_board_str;
            payload["boards"].push_back(item);
        }
        
        httplib::Client cli("127.0.0.1", 5000);
        cli.set_connection_timeout(5); 
        cli.set_read_timeout(30);
        
        // Hit the new endpoint
        auto res = cli.Post("/evaluate_regressor_only", payload.dump(), "application/json");
        
        if (res && res->status == 200) {
            json j_res = json::parse(res->body);
            for (size_t k = 0; k < solvable_for_regressor.size(); ++k) {
                size_t idx = solvable_for_regressor[k];
                (*surrogate_regressor_calls)++;
                double pushes = j_res[k]["pushes"];
                double branching = j_res[k]["branching"];
                
                double h_lb = population[idx].hungarian_lb;
                pushes = h_lb + std::clamp(pushes - h_lb, 0.0, 1.0 * h_lb);
                
                if (fitnessType == FitnessType::FO1_PUSHES || fitnessType == FitnessType::FO6_PUSHES_AND_SPEED) {
                    population[idx].fitness = pushes;
                } else if (fitnessType == FitnessType::FO2_ASTAR_EFF_BF || fitnessType == FitnessType::FO3_SOL_EFF_BF) {
                    population[idx].fitness = -branching;
                } else {
                    population[idx].fitness = pushes;
                }
            }
        } else {
            // Keep the A* fitness if server fails
            std::cerr << "Warning: /evaluate_regressor_only failed, using A* fitness.\n";
        }
        
        return;
    }

    // 1. Prepare JSON payload para candidatos dentro del régimen neuronal (< 6 cajas)
    json payload;
    payload["boards"] = json::array();
    
    for (size_t idx : surrogate_indices) {
        const auto& ind = population[idx];
        json item;
        item["board"] = board_to_pretty_string(ind.board);
        item["parent_board"] = ind.parent_board_str;
        payload["boards"].push_back(item);
    }

    // 2. Send HTTP POST request
    httplib::Client cli("127.0.0.1", 5000);
    cli.set_connection_timeout(5); // 5 seconds timeout
    cli.set_read_timeout(30);

    auto t_fs_flask_0 = std::chrono::high_resolution_clock::now();
    std::cerr << "[TIMING_PHASE] (b.2) POST Request a Flask (" << surrogate_indices.size() << " tableros)...\\n";
    auto res = cli.Post("/evaluate", payload.dump(), "application/json");
    auto t_fs_flask_1 = std::chrono::high_resolution_clock::now();
    std::cerr << "[TIMING_PHASE] (b.3) Respuesta Flask HTTP en: " << std::chrono::duration<double, std::milli>(t_fs_flask_1 - t_fs_flask_0).count() << " ms\\n";

    if (!res) {
        std::cerr << "Error: Failed to connect to Python Surrogate Server at 127.0.0.1:5000\n";
        std::cerr << "[TIMING_PHASE] (c.1) ALERTA: Fallback Silencioso a A* disparado (Conexion Fallida).\n";
        (*this->surrogate_fallbacks)++;

        auto original_heuristic = this->heuristic_type;
        auto original_max_sec = this->max_seconds;
        bool original_surrogate = this->use_surrogate;
        this->heuristic_type = Heuristic::hungarian;
        this->use_surrogate = false;
        this->max_seconds = 5.0;
        auto t_fb_start = std::chrono::high_resolution_clock::now();
        for (size_t idx : surrogate_indices) {
            evaluate(population[idx]);
        }
        auto t_fb_end = std::chrono::high_resolution_clock::now();
        std::cerr << "[TIMING_PHASE] (c.2) Ciclo A* de Fallback (batch entero) tardo: " << std::chrono::duration<double, std::milli>(t_fb_end - t_fb_start).count() << " ms\\n";
        this->heuristic_type = original_heuristic;
        this->use_surrogate = original_surrogate;
        this->max_seconds = original_max_sec;
        return;
    }

    if (res->status != 200) {
        std::cerr << "[TIMING_PHASE] (c.1) ALERTA: Fallback Silencioso a A* disparado (HTTP " << res->status << ").\n";
        (*this->surrogate_fallbacks)++;

        auto original_heuristic = this->heuristic_type;
        auto original_max_sec = this->max_seconds;
        bool original_surrogate = this->use_surrogate;
        this->heuristic_type = Heuristic::hungarian;
        this->use_surrogate = false;
        this->max_seconds = 5.0;
        for (size_t idx : surrogate_indices) {
            evaluate(population[idx]);
        }
        this->heuristic_type = original_heuristic;
        this->use_surrogate = original_surrogate;
        this->max_seconds = original_max_sec;
        return;
    }

    // 3. Parse JSON response
    try {
        json j_res = json::parse(res->body);
        
        for (size_t k = 0; k < surrogate_indices.size(); ++k) {
            size_t idx = surrogate_indices[k];
            bool is_solvable = j_res[k]["is_solvable"];
            
            if (!is_solvable) {
                population[idx].fitness = -1e9;
                (*classifier_deadlocks_filtered)++;
            } else {
                (*surrogate_regressor_calls)++;
                double pushes = j_res[k]["pushes"];
                double branching = j_res[k]["branching"];
                
                if (fitnessType == FitnessType::FO1_PUSHES) {
                    population[idx].fitness = pushes;
                } 
                else if (fitnessType == FitnessType::FO2_ASTAR_EFF_BF || fitnessType == FitnessType::FO3_SOL_EFF_BF) {
                    population[idx].fitness = -branching;
                }
                else {
                    population[idx].fitness = pushes;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << "\n";
        std::cerr << "Falling back to A* solver...\n";
        auto original_heuristic = this->heuristic_type;
        bool original_surrogate = this->use_surrogate;
        this->heuristic_type = Heuristic::hungarian;
        this->use_surrogate = false;
        for (size_t idx : surrogate_indices) {
            evaluate(population[idx]);
        }
        this->heuristic_type = original_heuristic;
        this->use_surrogate = original_surrogate;
    }
}

void Evaluator::filter_surrogate_batch(std::vector<Individual>& population)
{
    if (population.empty()) return;

    json payload;
    payload["boards"] = json::array();
    
    for (size_t i = 0; i < population.size(); ++i) {
        const auto& ind = population[i];
        json item;
        item["board"] = board_to_pretty_string(ind.board);
        item["parent_board"] = ind.parent_board_str;
        payload["boards"].push_back(item);
    }

    httplib::Client cli("127.0.0.1", 5000);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(30);

    auto res = cli.Post("/evaluate", payload.dump(), "application/json");

    if (!res || res->status != 200) {
        std::cerr << "[Classifier Filter] Warning: Surrogate server unavailable, proceeding with full A* evaluation.\n";
        return;
    }

    try {
        json j_res = json::parse(res->body);
        for (size_t k = 0; k < population.size(); ++k) {
            bool is_solvable = j_res[k]["is_solvable"];
            if (!is_solvable) {
                population[k].fitness = -1e9;
                (*this->classifier_deadlocks_filtered)++;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Classifier Filter] JSON error: " << e.what() << "\n";
    }
}

void Evaluator::evaluateDiagnostic(std::vector<Individual>& population, int generation) {
    std::cerr << "[DIAGNOSTIC] Running Regressor Diagnostic for Generation " << generation << "...\n";
    
    std::vector<Individual> pop_copy = population;
    
    auto original_heuristic = this->heuristic_type;
    bool original_surrogate = this->use_surrogate;
    double original_max_sec = this->max_seconds;
    
    this->use_surrogate = true;
    this->evaluate_surrogate_batch(pop_copy);
    
    std::vector<double> predicted_fitness(pop_copy.size());
    for (size_t i = 0; i < pop_copy.size(); i++) {
        predicted_fitness[i] = pop_copy[i].fitness;
    }
    
    this->use_surrogate = false;
    this->heuristic_type = Heuristic::hungarian;
    this->max_seconds = 15.0;
    
    for (size_t i = 0; i < pop_copy.size(); i++) {
        std::cerr << "[DIAGNOSTIC] Evaluating individual " << i+1 << "/" << pop_copy.size() << " with A* ground truth..." << std::flush;
        pop_copy[i].fitness = 0; // reset
        this->evaluate(pop_copy[i]);
        std::cerr << " Done. Actual Fit: " << pop_copy[i].fitness << ", Predicted: " << predicted_fitness[i] << "\n";
    }
    
    this->heuristic_type = original_heuristic;
    this->use_surrogate = original_surrogate;
    this->max_seconds = original_max_sec;
    
    std::ofstream out("scratch/regressor_diagnostic.csv", std::ios::app);
    out.seekp(0, std::ios::end);
    if (out.tellp() == 0) {
        out << "generation,candidate_idx,predicted_fitness,actual_fitness,is_solvable_prediction\n";
    }
    
    for (size_t i = 0; i < pop_copy.size(); i++) {
        bool is_solvable = (predicted_fitness[i] > -1e8);
        out << generation << "," << i << "," << predicted_fitness[i] << "," << pop_copy[i].fitness << "," << (is_solvable ? "True" : "False") << "\n";
    }
    out.close();
    
    std::cerr << "[DIAGNOSTIC] Finished Diagnostic for Generation " << generation << "\n";
}