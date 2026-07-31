#include "../../../include/evolution/evaluator/evaluator.h"
#include "../../../include/game_solver.h"
#include "../../../include/evolution/utils/board_utils.h"
#include <cmath>
#include <vector>
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

    // Cambia esto en tu Evaluator::evaluate para usar el matching perfecto O(n³) o neuronal
    auto stats = solver.test_template(Method::a_star, this->heuristic_type, solution, needs_path_simulator, nullptr, this->max_seconds);

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

    // 1. Prepare JSON payload
    json payload;
    payload["boards"] = json::array();
    
    for (const auto& ind : population) {
        json item;
        item["board"] = board_to_string(ind.board);
        item["parent_board"] = ind.parent_board_str;
        payload["boards"].push_back(item);
    }

    // 2. Send HTTP POST request
    httplib::Client cli("localhost", 5000);
    cli.set_connection_timeout(5); // 5 seconds timeout
    cli.set_read_timeout(30);

    auto res = cli.Post("/evaluate", payload.dump(), "application/json");

    if (!res) {
        std::cerr << "Error: Failed to connect to Python Surrogate Server at localhost:5000\n";
        std::cerr << "Falling back to A* solver for this batch...\n";
        
        // Ensure we fallback to Hungarian A* if surrogate server is down, so we don't try to load the JIT model in C++
        auto original_heuristic = this->heuristic_type;
        this->heuristic_type = Heuristic::hungarian;
        for (auto& ind : population) {
            evaluate(ind);
        }
        this->heuristic_type = original_heuristic;
        return;
    }

    if (res->status != 200) {
        std::cerr << "Error: Python Server returned HTTP " << res->status << "\n";
        std::cerr << "Response: " << res->body << "\n";
        
        auto original_heuristic = this->heuristic_type;
        this->heuristic_type = Heuristic::hungarian;
        for (auto& ind : population) {
            evaluate(ind);
        }
        this->heuristic_type = original_heuristic;
        return;
    }

    // 3. Parse JSON response
    try {
        json j_res = json::parse(res->body);
        
        for (size_t i = 0; i < population.size(); ++i) {
            bool is_solvable = j_res[i]["is_solvable"];
            
            if (!is_solvable) {
                population[i].fitness = -1e9;
            } else {
                double pushes = j_res[i]["pushes"];
                double branching = j_res[i]["branching"];
                
                // Asignamos el fitness dependiendo de lo que el usuario esté buscando
                if (fitnessType == FitnessType::FO1_PUSHES) {
                    population[i].fitness = pushes;
                } 
                else if (fitnessType == FitnessType::FO2_ASTAR_EFF_BF || fitnessType == FitnessType::FO3_SOL_EFF_BF) {
                    // Queremos ramas pequeñas, por lo que invertimos para que el EA que MAXIMIZA
                    // empuje el branching_factor hacia valores más bajos.
                    population[i].fitness = -branching;
                }
                else {
                    // Default fallback
                    population[i].fitness = pushes;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << "\n";
        std::cerr << "Falling back to A* solver...\n";
        auto original_heuristic = this->heuristic_type;
        this->heuristic_type = Heuristic::hungarian;
        for (auto& ind : population) {
            evaluate(ind);
        }
        this->heuristic_type = original_heuristic;
    }
}