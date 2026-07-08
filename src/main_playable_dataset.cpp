#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <algorithm>

#include "shell_generator/shell_generator.h"
#include "game_solver.h"
#include "evolution/evaluator/evaluator.h"
#include "evolution/utils/board_utils.h"

std::string serialize_board(const std::vector<std::vector<char>>& board) {
    std::string s = "";
    for (size_t i = 0; i < board.size(); i++) {
        for (size_t j = 0; j < board[i].size(); j++) {
            s += board[i][j];
        }
        if (i < board.size() - 1) s += "\n";
    }
    return s;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    int target_size = 7000;
    std::string output_file = "playable_dataset.csv";
    int custom_threads = -1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--target-size" && i + 1 < argc) {
            target_size = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            custom_threads = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "Starting Playable Dataset Generation...\n";
    std::cout << "Target size: " << target_size << "\n";
    std::cout << "Output file: " << output_file << "\n\n";
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    
    // Header
    file << "board_string,width,height,boxes,is_solvable,shell_hash,"
         << "runtime_ms,pushes,moves,generated_states,expanded_nodes,total_children,"
         << "effective_children,repeated_nodes,deadlocks,branching_real,"
         << "branching_effective,branching_classic,redundancy,closed_list_length,"
         << "path_branching_real_avg,path_branching_real_min,path_branching_real_max,"
         << "path_branching_effective_avg,path_branching_effective_min,path_branching_effective_max,"
         << "path_redundancy,path_deadlocks,path_box_lines,path_box_changes,solution_lurd\n";
    
    std::atomic<int> saved_count{0};
    std::vector<std::atomic<int>> box_counts(8); // Indices 1 to 7
    for (int i = 0; i < 8; i++) box_counts[i] = 0;
    
    std::mutex file_mutex;
    
    int per_box_target = target_size / 7;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto worker_task = [&]() {
        while (saved_count < target_size) {
            
            // Si ya llenamos todas las cuotas (por cuestiones de redondeo), break
            if (saved_count >= target_size) break;
            
            // Elegir num de cajas de forma uniforme pero respetando la cuota
            int target_box = 1 + (rand() % 7); // 1 a 7
            
            // Si este target ya cumplió su cuota, buscar otro que no
            if (box_counts[target_box] >= per_box_target) {
                bool found = false;
                for (int b = 1; b <= 7; b++) {
                    if (box_counts[b] < per_box_target) {
                        target_box = b;
                        found = true;
                        break;
                    }
                }
                // Si todos cumplieron la cuota base, se aceptan los extras para rellenar
                if (!found && saved_count >= 7 * per_box_target) {
                    target_box = 1 + (rand() % 7);
                } else if (!found) {
                    continue; 
                }
            }
            
            int factorX = 2 + (rand() % 3); // 2, 3 o 4
            int factorY = 2 + (rand() % 3); // 2, 3 o 4
            
            SokobanGenerator generator(factorX, factorY);
            generator.generate();
            std::vector<std::vector<char>> shell = generator.getBoard();
            
            auto deadlock_mask = compute_deadlock_mask(shell);
            int free_cells = count_free_cells(shell);
            
            // Necesitamos espacio para el jugador + 2*cajas (cajas + objetivos)
            if (free_cells < target_box * 2 + 1) continue;
            
            std::string shell_str = board_to_string(shell);
            size_t shell_hash = std::hash<std::string>{}(shell_str);
            
            // Colocar entidades
            auto board = shell;
            bool placement_success = true;
            
            try {
                placeRandom(board, '@', deadlock_mask);
                for (int k = 0; k < target_box; k++) {
                    placeRandom(board, '$', deadlock_mask);
                    placeRandom(board, '.', deadlock_mask);
                }
            } catch (const std::runtime_error& e) {
                placement_success = false;
            }
            
            if (!placement_success) continue;
            
            unsigned int rows = board.size();
            unsigned int cols = board[0].size();
            bool is_positive = false;
            SolverStats final_stats;
            
            {
                std::string level = board_to_string(board);
                
                game_solver solver(level, rows, cols, 32);
                std::vector<game_node> solution;
                
                // Habilitamos calc_path_branching = true
                final_stats = solver.test_template(Method::a_star, solution, true);
                if (final_stats.status == SolveStatus::TIMEOUT) continue;
                is_positive = (final_stats.status == SolveStatus::SOLVED);
            }
            
            if (is_positive) {
                std::string serialized = serialize_board(board);
                std::lock_guard<std::mutex> lock(file_mutex);
                
                // Doble chequeo dentro del lock
                if (saved_count < target_size) {
                    file << "\"" << serialized << "\"," 
                         << cols << "," << rows << "," << target_box << "," 
                         << 1 << "," << shell_hash << ","
                         << final_stats.runtime_ms << ","
                         << final_stats.pushes << ","
                         << final_stats.moves << ","
                         << final_stats.generated_states << ","
                         << final_stats.expanded_nodes << ","
                         << final_stats.total_children << ","
                         << final_stats.effective_children << ","
                         << final_stats.repeated_nodes << ","
                         << final_stats.deadlocks << ","
                         << final_stats.branching_real << ","
                         << final_stats.branching_effective << ","
                         << final_stats.branching_classic << ","
                         << final_stats.redundancy << ","
                         << final_stats.closed_list_length << ","
                         << final_stats.path_stats.get_branching_real_avg() << ","
                         << final_stats.path_stats.branching_real_min << ","
                         << final_stats.path_stats.branching_real_max << ","
                         << final_stats.path_stats.get_branching_effective_avg() << ","
                         << final_stats.path_stats.branching_effective_min << ","
                         << final_stats.path_stats.branching_effective_max << ","
                         << final_stats.path_stats.get_redundancy() << ","
                         << final_stats.path_stats.deadlocks << ","
                         << final_stats.path_stats.box_lines << ","
                         << final_stats.path_stats.box_changes << ",\""
                         << final_stats.lurd_path << "\"\n";
                    saved_count++;
                    box_counts[target_box]++;
                    
                    if (saved_count % 100 == 0) {
                        std::cout << "Progreso: " << saved_count << " / " << target_size << " | Cajas (";
                        for (int b = 1; b <= 7; b++) {
                            std::cout << b << ":" << box_counts[b] << " ";
                        }
                        std::cout << ")\n";
                    }
                }
            }
        }
    };
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (custom_threads > 0) num_threads = custom_threads;
    
    std::cout << "Using " << num_threads << " threads. Per-box target: " << per_box_target << "\n";
    
    std::vector<std::future<void>> futures;
    for (unsigned int t = 0; t < num_threads; t++) {
        futures.push_back(std::async(std::launch::async, worker_task));
    }
    
    for (auto& f : futures) {
        f.get();
    }
    
    file.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nDataset generation completed!\n";
    std::cout << "Time elapsed: " << duration.count() << " seconds.\n";
    std::cout << "Total Jugables: " << saved_count << "\n";
    for (int b = 1; b <= 7; b++) {
        std::cout << "Cajas " << b << ": " << box_counts[b] << "\n";
    }
    
    return 0;
}
