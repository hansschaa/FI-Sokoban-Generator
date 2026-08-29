#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <chrono>
#include <map>

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
    else if (fitness_arg == "FO6") fitnessType = FitnessType::FO6_PUSHES_AND_SPEED;
    else {
        std::cerr << "FO Invalida\n";
        return 1;
    }

    // 3. Variables Fijas y Estrictas para Experimento 1
    int maxEvals = 1000000;
    int stagLimit = 721;
    int maxCircuitTimeSeconds = 600; // Safe timeout for individual run
    if (char* val = getCmdOption(argv, argv + argc, "--maxEvals")) maxEvals = std::stoi(val);
    if (char* val = getCmdOption(argv, argv + argc, "--stagLimit")) stagLimit = std::stoi(val);
    if (char* val = getCmdOption(argv, argv + argc, "--timeLimit")) maxCircuitTimeSeconds = std::stoi(val);

    Heuristic heuristic_type = Heuristic::hungarian;
    if (char* val = getCmdOption(argv, argv + argc, "--heuristic")) {
        std::string h_arg = val;
        if (h_arg == "neural") heuristic_type = Heuristic::neural_batched;
        else if (h_arg == "neural_sequential") heuristic_type = Heuristic::neural;
        else if (h_arg == "hungarian") heuristic_type = Heuristic::hungarian;
        else if (h_arg == "simple") heuristic_type = Heuristic::simple;
        else if (h_arg == "hybrid_regressor") heuristic_type = Heuristic::hybrid_regressor;
        else if (h_arg == "classifier_filter") heuristic_type = Heuristic::classifier_filter;
        else if (h_arg == "full_surrogate") heuristic_type = Heuristic::full_surrogate;
    }
    
    double lambda_speed = 0.0;
    if (char* val = getCmdOption(argv, argv + argc, "--lambda_speed")) {
        lambda_speed = std::stod(val);
    }

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
    evaluator.use_surrogate = false; // Disable for initial generation!
    evaluator.max_seconds = 2.0; // Búsqueda ultrarrápida (máx 2s) para inicialización de semillas

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
                ind.parent_board_str = board_to_string(shell);
                
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

    auto t_start = std::chrono::steady_clock::now();

    // 6. Ejecución silenciosa de Metaheurísticas con Parseo Blindado
    try {
        if (algorithm == "ES")
        {
            EvolutionStrategy es;
            es.use_parallel = false; // IRace handles parallelism externally
            if (char* val = getCmdOption(argv, argv + argc, "--mu")) es.mu = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--lambda")) es.lambda = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) es.mutationRate = std::stod(val);

            es.maxEvaluations  = maxEvals;
            es.maxCircuitTimeSeconds = maxCircuitTimeSeconds;
            es.circuitStartTime = std::chrono::high_resolution_clock::now();
            es.stagnationLimit = (int)(7.14 * es.lambda);
            es.evaluator.fitnessType = fitnessType;
            es.evaluator.heuristic_type = heuristic_type;
            es.evaluator.use_surrogate = (heuristic_type != Heuristic::hungarian);
            
            std::cerr << "[IRACE] Effective params: maxEvals=" << es.maxEvaluations << ", stagLimit=" << es.stagnationLimit << ", timeLimit=" << es.maxCircuitTimeSeconds << std::endl;
            
            best = es.run(population);
        }
        else if (algorithm == "GA")
        {
            GeneticAlgorithm ga;
            ga.use_parallel = false; // IRace handles parallelism externally
            if (char* val = getCmdOption(argv, argv + argc, "--offspring")) ga.offspringSize = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) ga.maxFailedAttempts = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) ga.mutationRate = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--crossRate")) ga.crossoverRate = std::stod(val);

            ga.maxEvaluations  = maxEvals;
            ga.maxCircuitTimeSeconds = maxCircuitTimeSeconds;
            ga.circuitStartTime = std::chrono::high_resolution_clock::now();
            ga.stagnationLimit = (int)(7.14 * ga.offspringSize);
            ga.evaluator.fitnessType = fitnessType;
            ga.evaluator.heuristic_type = heuristic_type;
            ga.evaluator.use_surrogate = (heuristic_type != Heuristic::hungarian);

            best = ga.run(population);
        }
        else if (algorithm == "SA")
        {
            SimulatedAnnealing sa;
            
            // Valores por defecto en caso de ejecución manual
            sa.initialTemperature = 100.0;
            sa.coolingRate        = 0.01;
            sa.maxFailedAttempts  = 50;

            // PARSEO PARA IRACE
            if (char* val = getCmdOption(argv, argv + argc, "--initTemp")) sa.initialTemperature = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--coolRate")) sa.coolingRate = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) sa.maxFailedAttempts = std::stoi(val);

            sa.maxEvaluations     = maxEvals;   
            sa.stagnationLimit    = 200; // SA processes 1 individual per step, so 200 is fine
            sa.evaluator.fitnessType = fitnessType;
            sa.evaluator.heuristic_type = heuristic_type;
            sa.evaluator.use_surrogate = (heuristic_type != Heuristic::hungarian);

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
    auto t_end = std::chrono::steady_clock::now();
    double time_s = std::chrono::duration<double>(t_end - t_start).count();

    // 7. SALIDA ESTRICTA PARA IRACE (Solo el costo de minimización)
    if (fitnessType == FitnessType::FO6_PUSHES_AND_SPEED) {
        // Find base time
        std::string filename = board_file;
        size_t last_slash = filename.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            filename = filename.substr(last_slash + 1);
        }
        
        std::map<std::string, double> base_times = {
            {"shell_145.txt", 3.94},
            {"shell_15.txt", 5.30},
            {"shell_245.txt", 3.07},
            {"shell_577.txt", 38.87},
            {"shell_743.txt", 3.27}
        };
        
        if (base_times.find(filename) == base_times.end()) {
            std::cerr << "ERROR: FO6 requested but instance " << filename << " not found in hardcoded base_times map!\n";
            return 1;
        }
        
        double base_time = base_times[filename];
        double score = -best.fitness + lambda_speed * (time_s / base_time);
        std::cout << score << std::endl;
    } else {
        std::cout << -best.fitness << std::endl;
    }

    return 0;
}