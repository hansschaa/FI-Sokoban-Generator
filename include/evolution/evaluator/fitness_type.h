#pragma once

enum class FitnessType
{
    FO1_PUSHES,           // (baseline) Empujes mínimos
    FO2_ASTAR_EFF_BF,     // Branching factor efectivo promedio del A*
    FO3_SOL_EFF_BF,       // Branching factor efectivo a lo largo de la solución (Requiere Simulador)
    FO4_DEADLOCKS,        // Cantidad de estados deadlock encontrados en la resolución
    FO5_REPEATED_NODES    // Cantidad de nodos repetidos encontrados en la resolución
};