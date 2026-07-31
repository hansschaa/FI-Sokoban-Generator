#pragma once

#include "../individual.h"
#include "fitness_type.h"
#include <vector> // Incluido para soportar la matriz del tablero
#include "../../../include/game_solver.h"

class Evaluator
{
public:

    FitnessType fitnessType =
        FitnessType::FO1_PUSHES; // Valor por defecto, puede ser cambiado antes de evaluar
    Heuristic heuristic_type = Heuristic::hungarian; // Valor por defecto, puede ser cambiado antes de evaluar

    // Boolean flag to toggle between A* and Surrogate models
    bool use_surrogate = true;
    double max_seconds = 120.0;

    double evaluate(
        Individual& individual);

    // Evaluate an entire population using the Python Surrogate Server via HTTP
    void evaluate_surrogate_batch(std::vector<Individual>& population);

private:
    // Guarda el estado del tablero justo antes de enviarlo al solver A*
    void registrar_tablero_critico(const std::vector<std::vector<char>>& board);
};