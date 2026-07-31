#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <functional>

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

    int maxEvals = 1000;
    int stagLimit = 30;
    int maxCircuitTimeSeconds = -1;
    if (char* val = getCmdOption(argv, argv + argc, "--timeLimit")) {
        maxCircuitTimeSeconds = std::stoi(val);
    }
    if (char* val = getCmdOption(argv, argv + argc, "--maxEvals")) {
        maxEvals = std::stoi(val);
    }
    if (char* val = getCmdOption(argv, argv + argc, "--stagLimit")) {
        stagLimit = std::stoi(val);
    }

    Heuristic heuristic_type = Heuristic::hungarian;
    if (char* val = getCmdOption(argv, argv + argc, "--heuristic")) {
        std::string h_arg = val;
        if (h_arg == "neural") heuristic_type = Heuristic::neural_batched;
        else if (h_arg == "neural_sequential") heuristic_type = Heuristic::neural;
        else if (h_arg == "hungarian") heuristic_type = Heuristic::hungarian;
        else if (h_arg == "simple") heuristic_type = Heuristic::simple;
    }

    std::string out_csv_path = "";
    if (char* val = getCmdOption(argv, argv + argc, "--out_csv")) {
        out_csv_path = val;
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
            if (fitnessType == FitnessType::FO1_PUSHES) pop_size = 6;
            else if (fitnessType == FitnessType::FO4_DEADLOCKS) pop_size = 15;
            else if (fitnessType == FitnessType::FO5_REPEATED_NODES) pop_size = 15;
            else pop_size = 15; // default mu in ES
        }
    }

    std::vector<Individual> population;
    Evaluator evaluator;
    evaluator.fitnessType = fitnessType;

    // 5. Generar población inicial optimizada (Clonando el primer éxito)
    bool found_first = false;
    Individual first_valid;

    for (int i = 0; i < pop_size; i++)
    {
        if (found_first) {
            population.push_back(first_valid);
            continue;
        }

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
                    first_valid = ind;
                    valid = true;
                    found_first = true;
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
    
    // Calcular deadlock mask para el cascarón base
    std::vector<std::vector<bool>> deadlock_mask = compute_deadlock_mask(shell);

    // Setup logging
    std::shared_ptr<std::ofstream> csv_out;
    if (out_csv_path != "") {
        csv_out = std::make_shared<std::ofstream>(out_csv_path);
        if (csv_out->is_open()) {
            *csv_out << "time_ms,evaluations,fitness\n";
        }
    }
    
    auto log_progress = [csv_out](int evals, double best_fitness, double time_ms) {
        if (csv_out && csv_out->is_open()) {
            *csv_out << time_ms << "," << evals << "," << best_fitness << "\n";
            csv_out->flush();
        }
    };

    // 6. Ejecución silenciosa de Metaheurísticas con Parseo Blindado
    try {
        if (algorithm == "ES")
        {
            EvolutionStrategy es;
            es.setDeadlockMask(deadlock_mask);
            es.maxEvaluations  = maxEvals;
            es.stagnationLimit = stagLimit;
            es.evaluator.fitnessType = fitnessType;
            es.evaluator.heuristic_type = heuristic_type;
            es.evaluator.use_surrogate = (heuristic_type != Heuristic::hungarian);
            es.on_progress = log_progress;
            
            if (fitnessType == FitnessType::FO1_PUSHES) {
                es.mu = 6;
                es.lambda = 30;
                es.mutationRate = 0.9878;
            } else if (fitnessType == FitnessType::FO4_DEADLOCKS) {
                es.mu = 15;
                es.lambda = 20;
                es.mutationRate = 0.9569;
            } else if (fitnessType == FitnessType::FO5_REPEATED_NODES) {
                es.mu = 15;
                es.lambda = 20;
                es.mutationRate = 0.7987;
            }

            if (char* val = getCmdOption(argv, argv + argc, "--mu")) es.mu = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--lambda")) es.lambda = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) es.mutationRate = std::stod(val);

            es.maxCircuitTimeSeconds = maxCircuitTimeSeconds;
            es.circuitStartTime = std::chrono::high_resolution_clock::now();

            best = es.run(population);
        }
        else if (algorithm == "GA")
        {
            GeneticAlgorithm ga;
            ga.setDeadlockMask(deadlock_mask);
            ga.deadlock_mask = deadlock_mask; // For crossover
            ga.maxEvaluations  = maxEvals;
            ga.stagnationLimit = stagLimit;
            ga.evaluator.fitnessType = fitnessType;
            ga.evaluator.heuristic_type = heuristic_type;
            ga.evaluator.use_surrogate = (heuristic_type != Heuristic::hungarian);
            ga.on_progress = log_progress;
            
            if (fitnessType == FitnessType::FO1_PUSHES) {
                ga.offspringSize = 45;
                ga.maxFailedAttempts = 41;
                ga.mutationRate = 0.8339;
                ga.crossoverRate = 0.3729;
            } else if (fitnessType == FitnessType::FO4_DEADLOCKS) {
                ga.offspringSize = 35;
                ga.maxFailedAttempts = 14;
                ga.mutationRate = 0.9462;
                ga.crossoverRate = 0.9140;
            } else if (fitnessType == FitnessType::FO5_REPEATED_NODES) {
                ga.offspringSize = 33;
                ga.maxFailedAttempts = 18;
                ga.mutationRate = 0.7570;
                ga.crossoverRate = 0.8483;
            }

            if (char* val = getCmdOption(argv, argv + argc, "--offspring")) ga.offspringSize = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) ga.maxFailedAttempts = std::stoi(val);
            if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) ga.mutationRate = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--crossRate")) ga.crossoverRate = std::stod(val);
            
            ga.maxCircuitTimeSeconds = maxCircuitTimeSeconds;
            ga.circuitStartTime = std::chrono::high_resolution_clock::now();
            
            best = ga.run(population);
        }
        else if (algorithm == "SA")
        {
            SimulatedAnnealing sa;
            sa.moveMutation.deadlock_mask = deadlock_mask;
            sa.addMutation.deadlock_mask = deadlock_mask;
            sa.maxEvaluations     = maxEvals;   
            sa.stagnationLimit    = stagLimit;
            sa.evaluator.fitnessType = fitnessType;  
            sa.evaluator.heuristic_type = heuristic_type;
            sa.evaluator.use_surrogate = (heuristic_type != Heuristic::hungarian);
            sa.on_progress = log_progress;

            // Valores por defecto en caso de ejecución manual
            if (fitnessType == FitnessType::FO1_PUSHES) {
                sa.initialTemperature = 14.7079;
                sa.coolingRate = 0.0189;
                sa.maxFailedAttempts = 42;
            } else if (fitnessType == FitnessType::FO4_DEADLOCKS) {
                sa.initialTemperature = 28.0942;
                sa.coolingRate = 0.0344;
                sa.maxFailedAttempts = 66;
            } else if (fitnessType == FitnessType::FO5_REPEATED_NODES) {
                sa.initialTemperature = 601.5804;
                sa.coolingRate = 0.0368;
                sa.maxFailedAttempts = 77;
            } else {
                sa.initialTemperature = 100.0;
                sa.coolingRate        = 0.01;
                sa.maxFailedAttempts  = 50;
            }

            // PARSEO PARA IRACE
            if (char* val = getCmdOption(argv, argv + argc, "--initTemp")) sa.initialTemperature = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--coolRate")) sa.coolingRate = std::stod(val);
            if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) sa.maxFailedAttempts = std::stoi(val);
            
            sa.maxCircuitTimeSeconds = maxCircuitTimeSeconds;
            sa.circuitStartTime = std::chrono::high_resolution_clock::now();

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

    // 7. SALIDA ESTRICTA PARA EXPERIMENT (fitness;board_hash;board_string)
    std::string board_str = "";
    std::string board_str_flat = "";
    for (const auto& row : best.board) {
        for (char c : row) {
            board_str += c;
            board_str_flat += c;
        }
        board_str += "\n";
        board_str_flat += "|"; // Separador de filas
    }
    size_t board_hash = std::hash<std::string>{}(board_str);
    
    std::cout << -best.fitness << ";" << board_hash << ";" << board_str_flat << std::endl;

    return 0;
}