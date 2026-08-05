#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>
#include <fstream>
#include <utility>
#include <stdexcept>

#include "../include/evolution/algorithms/evolution_strategy.h"
#include "../include/evolution/algorithms/genetic_algorithm.h"
#include "../include/evolution/individual.h"
#include "../include/evolution/utils/board_utils.h"
#include "../include/game_solver.h"
#include "../include/evolution/algorithms/simulated_annealing.h"

std::vector<std::vector<char>> load_board(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("No se pudo abrir: " + filename);
    
    std::vector<std::string> lines;
    std::string line;
    size_t max_cols = 0;
    
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); 
        if (line.empty()) continue; 
        
        lines.push_back(line);
        if (line.size() > max_cols) {
            max_cols = line.size();
        }
    }
    
    if (lines.empty()) throw std::runtime_error("Error: El archivo del tablero esta vacio.");
    
    std::vector<std::vector<char>> board;
    for (const auto& l : lines) {
        std::vector<char> row(l.begin(), l.end());
        row.resize(max_cols, ' '); 
        board.push_back(row);
    }
    
    return board;
}

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        std::cerr << "Uso: ./irace_generator <ALGO> <FO> <SEED> <Ruta_Tablero> [parametros...]\n";
        return 1;
    }

    std::string algorithm  = argv[1];
    std::string fitness_arg = argv[2];
    std::string board_file = argv[4];

    // 1. Inicializar semilla dinámica para irace de forma blindada
    int seed = 0;
    try {
        seed = std::stoi(argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "Error parseando la semilla (argv[3]): '" << argv[3] << "'\n";
        return 1;
    }
    srand(seed);

    // 2. Parsear la Función Objetivo
    FitnessType fitnessType;
    if (fitness_arg == "FO1" || fitness_arg == "pushes") fitnessType = FitnessType::FO1_PUSHES;
    else if (fitness_arg == "FO2" || fitness_arg == "astar_bf") fitnessType = FitnessType::FO2_ASTAR_EFF_BF;
    else if (fitness_arg == "FO3" || fitness_arg == "sol_bf") fitnessType = FitnessType::FO3_SOL_EFF_BF;
    else if (fitness_arg == "FO4" || fitness_arg == "deadlocks") fitnessType = FitnessType::FO4_DEADLOCKS;
    else if (fitness_arg == "FO5" || fitness_arg == "repeated_nodes") fitnessType = FitnessType::FO5_REPEATED_NODES;
    else {
        std::cerr << "FO Invalida\n";
        return 1;
    }

    // 3. Variables Fijas y Estrictas para Experimento 1
    int maxEvals = 2000;
    int stagLimit = 200;

    // 4. Configurar el tablero inicial (Shell) y aplicar Flood Fill
    std::vector<std::vector<char>> shell;
    try {
        shell = load_board(board_file);
        
        int rows = shell.size();
        int cols = rows > 0 ? shell[0].size() : 0;
        std::vector<std::pair<int, int>> stack;
        
        for(int r = 0; r < rows; r++) { 
            stack.push_back(std::make_pair(r, 0)); 
            stack.push_back(std::make_pair(r, cols - 1)); 
        }
        for(int c = 0; c < cols; c++) { 
            stack.push_back(std::make_pair(0, c)); 
            stack.push_back(std::make_pair(rows - 1, c)); 
        }
        
        while(!stack.empty()) {
            int r = stack.back().first;
            int c = stack.back().second;
            stack.pop_back();
            
            if(r >= 0 && r < rows && c >= 0 && c < cols && shell[r][c] == ' ') {
                shell[r][c] = '#';
                stack.push_back(std::make_pair(r + 1, c));
                stack.push_back(std::make_pair(r - 1, c));
                stack.push_back(std::make_pair(r, c + 1));
                stack.push_back(std::make_pair(r, c - 1));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error en Flood Fill o parser: " << e.what() << "\n";
        return 1;
    }

    int pop_size = 10;
    if (algorithm == "ES") {
        if (char* val = getCmdOption(argv, argv + argc, "--mu")) {
            pop_size = std::stoi(val);
        } else {
            pop_size = 15; // default mu in ES
        }
    }

    std::vector<Individual> population;
    Evaluator evaluator;
    evaluator.fitnessType = fitnessType;

    // 5. Generar población inicial optimizada
    for (int i = 0; i < pop_size; i++)
    {
        bool valid = false;
        int attempts = 0;

        while (!valid && attempts < 50000) 
        {
            auto board = shell;
            
            try {
                placeRandom(board, '@');
                placeRandom(board, '$');
                placeRandom(board, '.');

                Individual ind;
                ind.board = board;
                
                double fit = evaluator.evaluate(ind);

                if (fit > -1e8)
                {
                    ind.fitness = fit;
                    population.push_back(ind);
                    valid = true;
                }
            } catch (...) {
                // Silenciar errores geométricos aleatorios y reintentar
            }
            attempts++;
        }

        if (!valid) {
            std::cerr << "Error: No se pudo generar un individuo valido tras 50,000 intentos." << std::endl;
            return 1;
        }
    }

    Individual best;

    // 6. Ejecución silenciosa de Metaheurísticas con Parseo Blindado
    try {
        if (algorithm == "ES")
        {
            EvolutionStrategy es;
            es.maxEvaluations  = maxEvals;
            es.stagnationLimit = stagLimit;
            es.evaluator.fitnessType = fitnessType;
            
            if (char* val = getCmdOption(argv, argv + argc, "--mu")) es.mu = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--lambda")) es.lambda = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) es.mutationRate = std::stod(val);

            best = es.run(population);
        }
        else if (algorithm == "GA")
        {
            GeneticAlgorithm ga;
            ga.maxEvaluations  = maxEvals;
            ga.stagnationLimit = stagLimit;
            ga.evaluator.fitnessType = fitnessType;
            
            if (char* val = getCmdOption(argv, argv + argc, "--offspring")) ga.offspringSize = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) ga.maxFailedAttempts = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) ga.mutationRate = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--crossRate")) ga.crossoverRate = std::stod(val);
            best = ga.run(population);
        }
        else if (algorithm == "SA")
        {
            SimulatedAnnealing sa;
            sa.maxEvaluations     = maxEvals;   
            sa.stagnationLimit    = 600;
            sa.evaluator.fitnessType = fitnessType;  

            // Valores por defecto en caso de ejecución manual
            sa.initialTemperature = 100.0;
            sa.coolingRate        = 0.01;
            sa.maxFailedAttempts  = 50;

            // PARSEO PARA IRACE
            if (char* val = getCmdOption(argv, argv + argc, "--initTemp")) sa.initialTemperature = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--coolRate")) sa.coolingRate = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) sa.maxFailedAttempts = std::stoi(val);

            Individual initial = *std::max_element(
                population.begin(), population.end(),
                [](const Individual& a, const Individual& b) {
                    return a.fitness < b.fitness;
                });
            best = sa.run(initial);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error critico parseando parametros (Posible texto en vez de numero): " << e.what() << "\n";
        return 1;
    }

    // 7. SALIDA ESTRICTA PARA IRACE (Solo el costo de minimización)
    std::cout << -best.fitness << std::endl;

    return 0;
}