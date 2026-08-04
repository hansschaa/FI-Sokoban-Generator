#include "../../../include/evolution/evaluator/evaluator.h"
#include "../../../include/game_solver.h"
#include "../../../include/evolution/utils/board_utils.h"
#include <cmath>
#include <vector>
#include <fstream>


void Evaluator::registrar_tablero_critico(const std::vector<std::vector<char>>& board) {
    // Disabled to avoid race conditions during multithreaded evaluation.
    /*
    std::ofstream out("tablero_actual.txt");
    if (out.is_open()) {
        for (const auto& row : board) {
            for (char c : row) {
                out << c;
            }
            out << "\n";
        }
        out.close();
    }
    */
}

double Evaluator::evaluate(Individual& individual)
{
    std::string level = board_to_string(individual.board);
    unsigned int rows = individual.board.size();
    unsigned int cols = individual.board[0].size();

    // Reducido a 64MB por hilo (antes 512MB) para prevenir Out Of Memory (OOM) en ejecución multihilo masiva.
    game_solver solver(level, rows, cols, 64);
    std::vector<game_node> solution;

    // MAGIA AQUI: Solo activamos el simulador de path si el fitness es FO3
    bool needs_path_simulator = (fitnessType == FitnessType::FO3_SOL_EFF_BF);

    // 1. Guardamos el tablero en el archivo temporal justo antes del peligro
    registrar_tablero_critico(individual.board);

    solver.enable_advanced_deadlocks = true;

    // Cambia esto en tu Evaluator::evaluate para usar el matching perfecto O(n³)
    auto stats = solver.test_template(Method::a_star, Heuristic::hungarian, solution, needs_path_simulator);

    if (stats.status != SolveStatus::SOLVED || stats.pushes <= 1)
    {
        if (stats.status == SolveStatus::TIMEOUT) {
            individual.censored = true;
        }
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