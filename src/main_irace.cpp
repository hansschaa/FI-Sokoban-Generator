#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>
#include <fstream>
#include <utility>

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
    
    // 1. Leer todas las líneas y buscar cuál es el ancho máximo
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // Limpiar Windows
        if (line.empty()) continue; // Ignorar líneas en blanco
        
        lines.push_back(line);
        if (line.size() > max_cols) {
            max_cols = line.size();
        }
    }
    
    if (lines.empty()) throw std::runtime_error("Error: El archivo del tablero esta vacio.");
    
    // 2. Construir la matriz garantizando que sea un rectángulo perfecto
    std::vector<std::vector<char>> board;
    for (const auto& l : lines) {
        std::vector<char> row(l.begin(), l.end());
        
        // RELLENO MÁGICO: Si la fila es más corta que max_cols, la rellena con espacios
        row.resize(max_cols, ' '); 
        
        board.push_back(row);
    }
    
    return board;
}

// Función auxiliar para leer parámetros inyectados por irace (ej: --maxEvals 3000)
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
    int seed = std::stoi(argv[3]);
    std::string board_file = argv[4];

    // 1. Inicializar semilla (Crítico para que irace evalúe de forma consistente)
    srand(seed);

    // 2. Parsear la Función Objetivo
    FitnessType fitnessType;
    if (fitness_arg == "FO1" || fitness_arg == "pushes") fitnessType = FitnessType::FO1_PUSHES;
    else if (fitness_arg == "FO2" || fitness_arg == "astar_bf") fitnessType = FitnessType::FO2_ASTAR_EFF_BF;
    else if (fitness_arg == "FO3" || fitness_arg == "sol_bf") fitnessType = FitnessType::FO3_SOL_EFF_BF;
    else if (fitness_arg == "FO4" || fitness_arg == "deadlocks") fitnessType = FitnessType::FO4_DEADLOCKS;
    else {
        std::cerr << "FO Invalida\n";
        return 1;
    }

    // 3. Leer parámetros dinámicos de calibración
    int maxEvals = 3000;
    int stagLimit = 200;

    if (char* val = getCmdOption(argv, argv + argc, "--maxEvals")) maxEvals = std::stoi(val);
    if (char* val = getCmdOption(argv, argv + argc, "--stagLimit")) stagLimit = std::stoi(val);

    // 4. Configurar el tablero inicial (Shell)
    std::vector<std::vector<char>> shell;
    try {
        shell = load_board(board_file);
        
        // --- INICIO DEL PARCHE: INUNDACIÓN EXTERIOR (FLOOD FILL) ---
        // Convierte el espacio muerto exterior en muros sólidos ('#')
        int rows = shell.size();
        int cols = rows > 0 ? shell[0].size() : 0;
        std::vector<std::pair<int, int>> stack;
        
        // 1. Agregar todos los bordes de la matriz a la pila
        for(int r = 0; r < rows; r++) { 
            stack.push_back(std::make_pair(r, 0)); 
            stack.push_back(std::make_pair(r, cols - 1)); 
        }
        for(int c = 0; c < cols; c++) { 
            stack.push_back(std::make_pair(0, c)); 
            stack.push_back(std::make_pair(rows - 1, c)); 
        }
        
        // 2. Pintar hacia adentro hasta chocar con los muros reales
        while(!stack.empty()) {
            int r = stack.back().first;
            int c = stack.back().second;
            stack.pop_back();
            
            // Si estamos dentro de los límites y es un espacio vacío...
            if(r >= 0 && r < rows && c >= 0 && c < cols && shell[r][c] == ' ') {
                shell[r][c] = '#'; // Lo solidificamos
                stack.push_back(std::make_pair(r + 1, c));
                stack.push_back(std::make_pair(r - 1, c));
                stack.push_back(std::make_pair(r, c + 1));
                stack.push_back(std::make_pair(r, c - 1));
            }
        }
        // --- FIN DEL PARCHE ---

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::vector<Individual> population;
    Evaluator evaluator;
    evaluator.fitnessType = fitnessType;
    const int POP_SIZE = 10;

    // 5. Generar población inicial
    std::cout << "DEBUG: Iniciando generacion de poblacion..." << std::endl;
    for (int i = 0; i < POP_SIZE; i++)
    {
        std::cout << "DEBUG: Generando individuo " << i << "..." << std::endl;
        bool valid = false;
        int attempts = 0;

        while (!valid && attempts < 100) // Reducido a 100 para evitar bucles infinitos
        {
            auto board = shell;
            
            try {
                // Colocación con control estricto
                placeRandom(board, '@');
                placeRandom(board, '$');
                placeRandom(board, '.');

                std::string level = board_to_string(board);
                unsigned int rows = board.size();
                unsigned int cols = rows > 0 ? board[0].size() : 0;

                std::cout << "  -> Creando solver..." << std::endl;
                game_solver solver(level, rows, cols, 512);
                
                std::vector<game_node> solution;
                
                std::cout << "  -> Resolviendo A* para este nivel:" << std::endl;
                std::cout << level << std::endl; // <-- ESTO ES LO NUEVO
                
                auto stats = solver.test_template(Method::a_star, solution);
                
                std::cout << "  -> Status devuelto: " << (int)stats.status << std::endl;

                if (stats.status == SolveStatus::SOLVED)
                {
                    Individual ind;
                    ind.board   = board;
                    
                    std::cout << "  -> Calculando fitness..." << std::endl;
                    ind.fitness = evaluator.evaluate(ind);
                    
                    population.push_back(ind);
                    valid = true;
                    std::cout << "DEBUG: Individuo " << i << " generado con exito." << std::endl;
                }
            } catch (const std::exception& e) {
                // Si placeRandom lanza el error de "TABLERO LLENO", lo capturamos aquí
                // Lo silenciamos temporalmente para no saturar la consola, a menos que sea error crítico
            }
            attempts++;
        }

        if (!valid) {
            std::cerr << "Error: No se pudo generar un individuo valido tras los intentos." << std::endl;
            return 1;
        }
    }
    Individual best;

    if (algorithm == "ES")
    {
        EvolutionStrategy es;
        es.maxEvaluations  = maxEvals;
        es.stagnationLimit = stagLimit;
        
        if (char* val = getCmdOption(argv, argv + argc, "--mu")) es.mu = std::stoi(val);
        if (char* val = getCmdOption(argv, argv + argc, "--lambda")) es.lambda = std::stoi(val);
        // NUEVO:
        if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) es.mutationRate = std::stod(val);

        best = es.run(population);
    }

    else if (algorithm == "GA")
    {
        GeneticAlgorithm ga;
        ga.maxEvaluations  = maxEvals;
        ga.stagnationLimit = stagLimit;
        
        if (char* val = getCmdOption(argv, argv + argc, "--offspring")) ga.offspringSize = std::stoi(val);
        if (char* val = getCmdOption(argv, argv + argc, "--maxFailed")) ga.maxFailedAttempts = std::stoi(val);
        // NUEVO:
        if (char* val = getCmdOption(argv, argv + argc, "--mutRate")) ga.mutationRate = std::stod(val);

        best = ga.run(population);
    }

    else if (algorithm == "SA")
    {
        SimulatedAnnealing sa;
        sa.initialTemperature = 100.0;
        sa.coolingRate        = 0.01;
        sa.maxEvaluations     = maxEvals;   // <- Inyectado
        sa.stagnationLimit    = stagLimit;  // <- Inyectado

        Individual initial = *std::max_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });
        best = sa.run(initial);
    }

    // 7. SALIDA ESTRICTA PARA IRACE (Solo el Costo)
    // Irace por defecto busca MINIMIZAR el output devuelto por consola. 
    // Como tus metaheurísticas están hechas para MAXIMIZAR, invertimos el signo aquí.
    // (Ej: Si el mejor fitness es 35 pushes, devolvemos -35 para que irace busque bajarlo más).
    std::cout << -best.fitness << std::endl;

    return 0;
}