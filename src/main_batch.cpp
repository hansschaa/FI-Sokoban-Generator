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

struct SokobanLevel {
    std::string name;
    std::vector<std::vector<char>> board;
};

// --- NUEVO: ALGORITMO FLOOD FILL PARA NORMALIZAR TABLEROS ---
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
    // Esto garantiza que TODO el exterior esté interconectado para el Flood Fill
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

// --- PARSER ACTUALIZADO ---
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
                normalize_board(current_level.board); // <--- APLICAMOS FLOOD FILL AQUI
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
        normalize_board(current_level.board); // <--- Y AQUI PARA EL ÚLTIMO TABLERO
        levels.push_back(current_level);
    }
    return levels;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Uso: ./batch_solver <coleccion.sok> <heuristica> <archivo_salida.txt> [calc]\n"
                  << "  heuristica: simple | hungarian\n"
                  << "  calc: opcional, 'calc' activa Path Branching Simulator desde el LURD\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string heuristic_arg = argv[2];
    std::string output_file = argv[3];

    Heuristic heuristic_type;
    if (heuristic_arg == "hungarian") heuristic_type = Heuristic::hungarian;
    else if (heuristic_arg == "simple") heuristic_type = Heuristic::simple;
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

    std::vector<SokobanLevel> collection;
    try {
        collection = load_sok_collection(input_file);
        std::cout << "🚀 Coleccion cargada. Procesando " << collection.size() << " tableros...\n";
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

    // Cabecera TSV completa con todas tus métricas de búsqueda y del simulador LURD
    out << "LevelName\tStatus\tRuntime_ms\tPushes\tMoves\tGeneratedStates\tExpandedNodes\t"
        << "TotalChildren\tEffectiveChildren\tRepeatedNodes\tDeadlocks\tBranchingReal\tBranchingEffective\tBranchingClassic\tRedundancy\t"
        << "PathStates\tPathBoxLines\tPathBoxChanges\tPathBranchingRealAvg\tPathBranchingEffectiveAvg\n";

    int idx = 1;
    for (const auto& lvl : collection) {
        std::cout << "[" << idx++ << "/" << collection.size() << "] " 
                  << lvl.name.substr(0, 40) << "... " << std::flush;

        std::string level_str = board_to_string(lvl.board);
        unsigned int rows = lvl.board.size();
        unsigned int cols = lvl.board.empty() ? 0 : lvl.board[0].size();

        // Instanciamos el solver pasándole la variable lvalue string_str obligatoria por referencia
        game_solver solver(level_str, rows, cols, 512);
        std::vector<game_node> solution;

        // Llama al solver directamente
        auto stats = solver.test_template(Method::a_star, heuristic_type, solution, calc_path_branching);

        // En lugar de calcular el "duration_ms" afuera, usas la métrica interna rigurosa
        double duration_ms = stats.runtime_ms;

        std::string status_str = (stats.status == SolveStatus::SOLVED) ? "SOLVED" :
                                 (stats.status == SolveStatus::TIMEOUT) ? "TIMEOUT" : "UNSOLVABLE";

        std::cout << status_str << " (" << std::fixed << std::setprecision(2) << duration_ms << " ms)\n";

        // Escritura de la fila de datos estructurados para Pandas
        out << lvl.name << "\t"
            << status_str << "\t"
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
            << stats.redundancy << "\t";

        // Si se calculó el Path LURD Simulator, guardamos sus desgloses; si no, dejamos valores vacíos o 0
        if (stats.path_stats_calculated) {
            out << stats.path_stats.states << "\t"
                << stats.path_stats.box_lines << "\t"
                << stats.path_stats.box_changes << "\t"
                << stats.path_stats.get_branching_real_avg() << "\t"
                << stats.path_stats.get_branching_effective_avg() << "\n";
        } else {
            out << "0\t0\t0\t0.0\t0.0\n";
        }
    }

    out.close();
    std::cout << "\n✅ Archivo tabular generado con exito: " << output_file << "\n";
    return 0;
}