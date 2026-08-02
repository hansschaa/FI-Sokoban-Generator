#pragma once

#include "../individual.h"
#include "fitness_type.h"
#include <vector> // Incluido para soportar la matriz del tablero
#include "../../../include/game_solver.h"
#include <memory>

class Evaluator
{
public:

    FitnessType fitnessType =
        FitnessType::FO1_PUSHES; // Valor por defecto, puede ser cambiado antes de evaluar
    Heuristic heuristic_type = Heuristic::hungarian; // Valor por defecto, puede ser cambiado antes de evaluar

    // Boolean flag to toggle between A* and Surrogate models
    bool use_surrogate = true;
    double max_seconds = 120.0;
    std::shared_ptr<int> surrogate_fallbacks = std::make_shared<int>(0);
    std::shared_ptr<int> surrogate_regressor_calls = std::make_shared<int>(0);
    std::shared_ptr<int> hybrid_hungarian_delegations = std::make_shared<int>(0);
    std::shared_ptr<int> classifier_deadlocks_filtered = std::make_shared<int>(0);
    std::shared_ptr<int> classifier_false_positives = std::make_shared<int>(0);

    double evaluate(
        Individual& individual);

    // Evaluate an entire population using the Python Surrogate Server via HTTP
    void evaluate_surrogate_batch(std::vector<Individual>& population);

    // Fast pre-solver filter using only the contrastive classifier via HTTP
    void filter_surrogate_batch(std::vector<Individual>& population);

    // Diagnostics: save surrogate predictions and ground truth A* to CSV
    void evaluateDiagnostic(std::vector<Individual>& population, int generation);

private:
    // Guarda el estado del tablero justo antes de enviarlo al solver A*
    void registrar_tablero_critico(const std::vector<std::vector<char>>& board);
};