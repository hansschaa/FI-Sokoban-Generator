#pragma once

#include "../individual.h"
#include "fitness_type.h"

class Evaluator
{
public:

    FitnessType fitnessType =
        FitnessType::FO1_PUSHES; // Valor por defecto, puede ser cambiado antes de evaluar

    double evaluate(
        Individual& individual);
};