#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <queue>

#include "../include/evolution/algorithms/evolution_strategy.h"
#include "../include/evolution/algorithms/genetic_algorithm.h"
#include "../include/evolution/algorithms/simulated_annealing.h"

#include "../include/evolution/individual.h"
#include "../include/evolution/utils/board_utils.h"
#include "../include/game_solver.h"

// --- ALGORITMO FLOOD FILL PARA NORMALIZAR TABLEROS ---
void normalize_board(std::vector<std::vector<char>>& board) {
    if (board.empty()) return;
    
    // 1. Encontrar el ancho máximo de la matriz irregular
    size_t max_cols = 0;
    for (const auto& row : board) {
        max_cols = std::max(max_cols, row.size());
    }
    
    // 2. Rellenar con espacios vacíos para hacer un rectángulo perfecto
    for (auto& row : board) {
        while (row.size() < max_cols) row.push_back(' ');
    }
    
    // 3. Añadir un "Borde de Seguridad" de 1 celda alrededor de todo el mapa
    max_cols += 2;
    for (auto& row : board) {
        row.insert(row.begin(), ' ');
        row.push_back(' ');
    }
    board.insert(board.begin(), std::vector<char>(max_cols, ' '));
    board.push_back(std::vector<char>(max_cols, ' '));
    
    int rows = board.size();
    int cols = max_cols;
    
    // 4. Flood fill (BFS) desde (0,0) (que sabemos que es 100% exterior)
    std::queue<std::pair<int, int>> q;
    q.push({0, 0});
    board[0][0] = '#'; // Sellar el inicio
    
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                // Inundar solo si es espacio vacío exterior o un guión
                if (board[nr][nc] == ' ' || board[nr][nc] == '-' || board[nr][nc] == '_') {
                    board[nr][nc] = '#';
                    q.push({nr, nc});
                }
            }
        }
    }
    
    // 5. Bounding Box: Recortar exceso de muros '#' para que el A* no itere de más
    int min_r = rows, max_r = 0, min_c = cols, max_c = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (board[r][c] != '#') { // Rastrear la "zona jugable"
                min_r = std::min(min_r, r);
                max_r = std::max(max_r, r);
                min_c = std::min(min_c, c);
                max_c = std::max(max_c, c);
            }
        }
    }
    
    // Expandir 1 celda para conservar la capa de paredes perimetrales
    min_r = std::max(0, min_r - 1);
    max_r = std::min(rows - 1, max_r + 1);
    min_c = std::max(0, min_c - 1);
    max_c = std::min(cols - 1, max_c + 1);
    
    std::vector<std::vector<char>> cropped;
    for (int r = min_r; r <= max_r; ++r) {
        std::vector<char> new_row;
        for (int c = min_c; c <= max_c; ++c) {
            new_row.push_back(board[r][c]);
        }
        cropped.push_back(new_row);
    }
    
    board = cropped;
}

//
// FASE 1: ARRAY ESTÁTICO DE 30 SEMILLAS
//
static const int EXPERIMENT_SEEDS[30] = {
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
    1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030
};

// Parser para levels/experiments_shells.txt
std::vector<std::vector<std::vector<char>>> load_shells(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo de shells: " + filename);
    }

    std::vector<std::vector<std::vector<char>>> shells;
    std::string line;
    std::vector<std::vector<char>> current_shell;
    bool reading = false;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.find("Shell ID:") != std::string::npos) {
            if (!current_shell.empty()) {
                normalize_board(current_shell);
                shells.push_back(current_shell);
                current_shell.clear();
            }
            reading = true;
        } else if (line.find("=========") != std::string::npos) {
            reading = false;
        } else if (reading) {
            // Si la linea no está vacía y solo tiene '#' o ' '
            if (!line.empty() && line.find_first_not_of(" #") == std::string::npos && line.find('#') != std::string::npos) {
                current_shell.emplace_back(line.begin(), line.end());
            }
        }
    }
    if (!current_shell.empty()) {
        normalize_board(current_shell);
        shells.push_back(current_shell);
    }
    return shells;
}

int main(int argc, char** argv)
{
    // Uso: ./experiment_1 <fo> <bt_id> <algoritmo> <run_index>
    // fo: Pushes | Deadlocks | NodosRepetidos
    // bt_id: 1 a 20
    // algoritmo: SA | ES | GA
    // run_index: 1 a 30

    if (argc < 5)
    {
        std::cerr << "Uso: ./experiment_1 <FO> <BT_id> <Algoritmo> <Run_Index>\n";
        return 1;
    }

    std::string fo_str  = argv[1];
    int bt_id           = std::stoi(argv[2]);
    std::string alg_str = argv[3];
    int run_index       = std::stoi(argv[4]);

    if (bt_id < 1 || bt_id > 20) {
        std::cerr << "Error: BT_id debe estar entre 1 y 20.\n";
        return 1;
    }
    if (run_index < 1 || run_index > 30) {
        std::cerr << "Error: Run_Index debe estar entre 1 y 30.\n";
        return 1;
    }

    // 1. FORZAR LA SEMILLA DEL GENERADOR ALEATORIO
    int seed = EXPERIMENT_SEEDS[run_index - 1];
    srand(seed);

    // 2. CONFIGURAR LA FUNCIÓN OBJETIVO
    FitnessType fitnessType;
    if (fo_str == "Pushes") {
        fitnessType = FitnessType::FO1_PUSHES;
    } else if (fo_str == "Deadlocks") {
        fitnessType = FitnessType::FO4_DEADLOCKS;
    } else if (fo_str == "NodosRepetidos") {
        fitnessType = FitnessType::FO5_REPEATED_NODES;
    } else {
        std::cerr << "Error: FO desconocida '" << fo_str << "'. Opciones: Pushes, Deadlocks, NodosRepetidos.\n";
        return 1;
    }

    // 3. INSTANCIAR EL TABLERO BASE (SHELL)
    std::vector<std::vector<std::vector<char>>> shells;
    try {
        shells = load_shells("levels/experiments_shells.txt");
    } catch(const std::exception& e) {
        std::cerr << "Excepcion cargando shells: " << e.what() << "\n";
        return 1;
    }

    if (shells.size() < 20) {
        std::cerr << "Error: Se esperaban al menos 20 shells, se cargaron " << shells.size() << ".\n";
        return 1;
    }

    auto shell = shells[bt_id - 1];

    // 4. GENERAR POBLACIÓN INICIAL Y EJECUTAR METAHEURÍSTICA
    std::vector<Individual> population;
    Evaluator evaluator;
    evaluator.fitnessType = fitnessType;

    const int POP_SIZE = 10;
    for (int i = 0; i < POP_SIZE; i++)
    {
        bool valid = false;
        int attempts = 0;

        while (!valid && attempts < 10000)
        {
            auto board = shell;
            int numBoxes = 1;

            placeRandom(board, '@');
            for (int k = 0; k < numBoxes; k++)
            {
                placeRandom(board, '$');
                placeRandom(board, '.');
            }

            std::string level = board_to_string(board);
            unsigned int rows = board.size();
            unsigned int cols = board.empty() ? 0 : board[0].size();

            game_solver solver(level, rows, cols, 512);
            std::vector<game_node> solution;
            auto stats = solver.test_template(Method::a_star, solution);

            if (stats.status == SolveStatus::SOLVED)
            {
                Individual ind;
                ind.board = board;
                ind.fitness = evaluator.evaluate(ind);
                population.push_back(ind);
                valid = true;
            }
            attempts++;
        }

        if (!valid) {
            std::cerr << "Error: No se pudo generar el individuo inicial " << i << ".\n";
            return 1;
        }
    }

    Individual best;

    if (alg_str == "ES")
    {
        EvolutionStrategy es;
        es.mu              = 5;
        es.lambda          = 10;
        es.maxEvaluations  = 500;
        es.stagnationLimit = 5;
        // Ocultar prints para no ensuciar el script, aunque los logs estan dirigidos al archivo.
        best = es.run(population);
    }
    else if (alg_str == "GA")
    {
        GeneticAlgorithm ga;
        ga.offspringSize   = 10;
        ga.maxEvaluations  = 500;
        ga.stagnationLimit = 15;
        best = ga.run(population);
    }
    else if (alg_str == "SA")
    {
        SimulatedAnnealing sa;
        sa.initialTemperature = 100.0;
        sa.coolingRate        = 0.01;
        sa.maxEvaluations     = 500;
        sa.stagnationLimit    = 15;

        Individual initial = *std::max_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });
        best = sa.run(initial);
    }
    else
    {
        std::cerr << "Error: Algoritmo desconocido '" << alg_str << "'. Opciones: SA, ES, GA.\n";
        return 1;
    }

    // 5. IMPRIMIR EL MEJOR FITNESS DE LA ÚLTIMA ITERACIÓN
    // Formato CSV (Tidy): FO, BT_id, Algoritmo, Semilla_ID, Fitness_Bruto
    
    // NOTA: Para NodosRepetidos, guardamos el fitness en Evaluator de forma negativa para minimizar.
    // Aquí lo volvemos a invertir para guardar el número bruto real en el CSV.
    double fitness_bruto = best.fitness;
    if (fo_str == "NodosRepetidos") {
        fitness_bruto = -fitness_bruto;
    }

    // Preparar formato del string de BT_id (ej. "BT_01", "BT_14")
    std::string bt_str = "BT_";
    if (bt_id < 10) bt_str += "0";
    bt_str += std::to_string(bt_id);

    // Escribir al archivo
    std::ofstream out("exp1_raw_data.csv", std::ios_base::app);
    if (!out.is_open()) {
        std::cerr << "Error: No se pudo abrir exp1_raw_data.csv para escribir.\n";
        return 1;
    }

    out << fo_str << "," 
        << bt_str << "," 
        << alg_str << "," 
        << run_index << "," 
        << fitness_bruto << "\n";
        
    out.close();

    return 0;
}
