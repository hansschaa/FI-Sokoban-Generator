#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <queue> // <- CRÍTICO PARA EL FLOOD FILL

#include "../include/game_solver.h"
#include "../include/evolution/utils/board_utils.h"
#include "neural_heuristic.hpp"

struct SokobanLevel {
    std::string name;
    std::vector<std::vector<char>> board;
};

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

// --- PARSER ---
std::vector<SokobanLevel> load_sok_collection(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir la coleccion: " + filename);
    }

    std::vector<SokobanLevel> levels;
    std::string line;
    SokobanLevel current_level;
    bool reading_board = false;

    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        bool is_board_line = !line.empty() && line.find_first_not_of("# @$.*+-") == std::string::npos;

        if (!is_board_line) {
            if (reading_board) {
                normalize_board(current_level.board);
                levels.push_back(current_level);
                current_level = SokobanLevel();
                reading_board = false;
            }
            current_level.name = line;
        } else {
            current_level.board.emplace_back(line.begin(), line.end());
            reading_board = true;
        }
    }
    if (reading_board) {
        normalize_board(current_level.board);
        levels.push_back(current_level);
    }
    return levels;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Uso: ./batch_solver <coleccion.sok> <heuristica> <archivo_salida.txt> [calc] [advanced]\n"
                  << "  heuristica: simple | hungarian\n"
                  << "  calc: opcional, 'calc' activa Path Branching Simulator desde el LURD\n"
                  << "  advanced: opcional, 'advanced' o 'true' activa la deteccion avanzada de deadlocks (freeze + bipartito)\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string heuristic_arg = argv[2];
    std::string output_file = argv[3];

    Heuristic heuristic_type;
    if (heuristic_arg == "hungarian") heuristic_type = Heuristic::hungarian;
    else if (heuristic_arg == "simple") heuristic_type = Heuristic::simple;
    else if (heuristic_arg == "manhattan") heuristic_type = Heuristic::manhattan;
    else if (heuristic_arg == "neural") heuristic_type = Heuristic::neural_batched; // Default to batched for max speed
    else if (heuristic_arg == "neural_sequential") heuristic_type = Heuristic::neural; // Para el ablation study
    else if (heuristic_arg == "neural_batched") heuristic_type = Heuristic::neural_batched;
    else if (heuristic_arg == "neural_batched_massive") heuristic_type = Heuristic::neural_batched_massive;
    else {
        std::cerr << "Heuristica desconocida: " << heuristic_arg << "\n";
        return 1;
    }

    bool calc_path_branching = false;
    if (argc >= 5) {
        std::string flag = argv[4];
        if (flag == "calc" || flag == "1" || flag == "true") {
            calc_path_branching = true;
        }
    }

    bool enable_advanced = false;
    if (argc >= 6) {
        std::string flag = argv[5];
        if (flag == "advanced" || flag == "1" || flag == "true") {
            enable_advanced = true;
        }
    }

    std::vector<SokobanLevel> collection;
    try {
        collection = load_sok_collection(input_file);
        std::cout << "🚀 Coleccion cargada. Procesando " << collection.size() << " tableros con heuristica [" << heuristic_arg << "]...\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "No se pudo crear el archivo de salida.\n";
        return 1;
    }

    // --- NUEVA CABECERA TSV CON TODAS LAS MÉTRICAS ---
    out << "LevelName\tStatus\tLURD_Path\tRuntime_ms\tPushes\tMoves\t"
        << "GeneratedStates\tExpandedNodes\tTotalChildren\tEffectiveChildren\tRepeatedNodes\tDeadlocks\t"
        << "BranchingReal\tBranchingEffective\tBranchingClassic\tRedundancy\tClosedListLength\t"
        << "PathStates\tPathBoxLines\tPathBoxChanges\t"
        << "PathBranchingRealTotalNodes\tPathBranchingRealMin\tPathBranchingRealMax\tPathBranchingRealAvg\t"
        << "PathBranchingEffectiveTotalNodes\tPathBranchingEffectiveMin\tPathBranchingEffectiveMax\tPathBranchingEffectiveAvg\t"
        << "PathTotalChildrenGenerated\tPathRepeatedNodes\tPathDeadlocks\tPathRedundancy\n";

    std::shared_ptr<NeuralHeuristic> shared_net = nullptr;
    if (heuristic_type == Heuristic::neural ||
        heuristic_type == Heuristic::neural_batched ||
        heuristic_type == Heuristic::neural_batched_massive) {
        std::string model_path = "surrogate_models/results/surrogate_regressor_jit.pt";
        if (const char* env_p = std::getenv("MODEL_PATH")) {
            model_path = env_p;
        }
        shared_net = std::make_shared<NeuralHeuristic>(model_path, 25, 25);
    }

    int idx = 1;
    for (const auto& lvl : collection) {
        std::cout << "[" << idx++ << "/" << collection.size() << "] " 
                  << lvl.name.substr(0, 40) << "... " << std::flush;

        std::string level_str = board_to_string(lvl.board);
        unsigned int rows = lvl.board.size();
        unsigned int cols = lvl.board.empty() ? 0 : lvl.board[0].size();

        game_solver solver(level_str, rows, cols, 4096);
        solver.enable_advanced_deadlocks = enable_advanced;
        std::vector<game_node> solution;
        
        double max_secs = 120.0;
        if (const char* env_timeout = std::getenv("MAX_SECONDS")) {
            max_secs = std::stod(env_timeout);
        }

        auto stats = solver.test_template(Method::a_star, heuristic_type, solution, calc_path_branching, shared_net, max_secs, 100000000);
        double duration_ms = stats.runtime_ms;

        std::string status_str = (stats.status == SolveStatus::SOLVED) ? "SOLVED" :
                                 (stats.status == SolveStatus::TIMEOUT) ? "TIMEOUT" :
                                 (stats.status == SolveStatus::OOM) ? "OOM" : "UNSOLVABLE";

        std::cout << status_str << " | " << std::fixed << std::setprecision(2) << duration_ms << " ms | Nodos Expandidos: " << stats.expanded_nodes << "\n";

        // --- EXPORTACIÓN DE TODAS LAS VARIABLES BASE ---
        out << lvl.name << "\t"
            << status_str << "\t"
            << (stats.lurd_path.empty() ? "NONE" : stats.lurd_path) << "\t"
            << duration_ms << "\t"
            << stats.pushes << "\t"
            << stats.moves << "\t"
            << stats.generated_states << "\t"
            << stats.expanded_nodes << "\t"
            << stats.total_children << "\t"
            << stats.effective_children << "\t"
            << stats.repeated_nodes << "\t"
            << stats.deadlocks << "\t"
            << stats.branching_real << "\t"
            << stats.branching_effective << "\t"
            << stats.branching_classic << "\t"
            << stats.redundancy << "\t"
            << stats.closed_list_length << "\t";

        // --- EXPORTACIÓN DEL SIMULADOR LURD EXPANDIDO ---
        if (stats.path_stats_calculated) {
            out << stats.path_stats.states << "\t"
                << stats.path_stats.box_lines << "\t"
                << stats.path_stats.box_changes << "\t"
                << stats.path_stats.branching_real_total_nodes << "\t"
                << stats.path_stats.branching_real_min << "\t"
                << stats.path_stats.branching_real_max << "\t"
                << stats.path_stats.get_branching_real_avg() << "\t"
                << stats.path_stats.branching_effective_total_nodes << "\t"
                << stats.path_stats.branching_effective_min << "\t"
                << stats.path_stats.branching_effective_max << "\t"
                << stats.path_stats.get_branching_effective_avg() << "\t"
                << stats.path_stats.total_children_generated << "\t"
                << stats.path_stats.repeated_nodes << "\t"
                << stats.path_stats.deadlocks << "\t"
                << stats.path_stats.get_redundancy() << "\n";
        } else {
            // Rellenar las 15 columnas del simulador con ceros si no se calculó
            out << "0\t0\t0\t"
                << "0\t0.0\t0.0\t0.0\t"
                << "0\t0.0\t0.0\t0.0\t"
                << "0\t0\t0\t0.0\n";
        }
    }

    out.close();
    std::cout << "\n✅ Archivo tabular generado con exito: " << output_file << "\n";
    return 0;
}