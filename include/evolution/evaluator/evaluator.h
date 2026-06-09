#pragma once

#include "../individual.h"
#include "fitness_type.h"
#include <vector> // Incluido para soportar la matriz del tablero

class Evaluator
{
public:

    FitnessType fitnessType =
        FitnessType::FO1_PUSHES; // Valor por defecto, puede ser cambiado antes de evaluar

    double evaluate(
        Individual& individual);

private:
    // Guarda el estado del tablero justo antes de enviarlo al solver A*
    void registrar_tablero_critico(const std::vector<std::vector<char>>& board);
};