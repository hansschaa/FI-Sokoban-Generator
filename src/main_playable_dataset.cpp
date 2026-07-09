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
#include <cmath>

#include "shell_generator/shell_generator.h"
#include "game_solver.h"
#include "evolution/evaluator/evaluator.h"
#include "evolution/utils/board_utils.h"
#include "structural_metrics.h"

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
         << "path_redundancy,path_deadlocks,path_box_lines,path_box_changes,solution_lurd,"
         << "wall_density,open_space_ratio,connectivity,aspect_ratio,dead_end_ratio,"
         << "avg_symmetry,num_interior_regions,initial_optimal_distance\n";
    
    std::atomic<int> saved_count{0};
    std::vector<std::atomic<int>> box_counts(7); // Indices 1 to 6
    for (int i = 0; i < 7; i++) box_counts[i] = 0;
    
    std::mutex file_mutex;
    
    int per_box_target = std::ceil(target_size / 6.0);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto worker_task = [&]() {
        while (saved_count < target_size) {
            
            if (saved_count >= target_size) break;
            
            // Buscar la clase de caja secuencialmente
            int target_box = 1;
            for (int b = 1; b <= 6; b++) {
                if (box_counts[b].load() < per_box_target) {
                    target_box = b;
                    break;
                }
            }
            if (box_counts[target_box].load() >= per_box_target) {
                int min_count = box_counts[1].load();
                target_box = 1;
                for (int b = 2; b <= 6; b++) {
                    if (box_counts[b].load() < min_count) {
                        min_count = box_counts[b].load();
                        target_box = b;
                    }
                }
            }
            
            // Random shell dimensions (2 to 4)
            int factorX = 2 + (rand() % 3);
            int factorY = 2 + (rand() % 3);
            
            SokobanGenerator generator(factorX, factorY);
            generator.generate();
            std::vector<std::vector<char>> shell = generator.getBoard();
            
            auto deadlock_mask = compute_deadlock_mask(shell);
            int free_cells = count_free_cells(shell);
            
            // Need at least space for target_box
            if (free_cells < target_box * 2 + 1) continue;
            
            std::string shell_str = board_to_string(shell);
            size_t shell_hash = std::hash<std::string>{}(shell_str);
            
            auto current_board = shell;
            
            try {
                placeRandom(current_board, '@', deadlock_mask);
            } catch (...) {
                continue;
            }
            
            int current_boxes = 0;
            bool reached_target = false;
            SolverStats final_stats_for_target;
            
            while (current_boxes < target_box) {
                if (saved_count >= target_size) break;
                
                bool found_next = false;
                
                // Intentar hasta 50 veces agregar una caja
                for (int attempt = 0; attempt < 50; attempt++) {
                    if (saved_count >= target_size) break;
                    
                    auto temp_board = current_board;
                    try {
                        placeRandom(temp_board, '$', deadlock_mask);
                        placeRandom(temp_board, '.', deadlock_mask);
                    } catch (...) {
                        continue;
                    }
                    
                    unsigned int rows = temp_board.size();
                    unsigned int cols = temp_board[0].size();
                    
                    std::string level = board_to_string(temp_board);
                    game_solver solver(level, rows, cols, 32);
                    std::vector<game_node> solution;
                    
                    SolverStats stats = solver.test_template(Method::a_star, solution, true);
                    if (stats.status == SolveStatus::TIMEOUT) continue;
                    
                    if (stats.status == SolveStatus::SOLVED) {
                        found_next = true;
                        current_board = temp_board; // Commit the new box
                        if (current_boxes + 1 == target_box) {
                            final_stats_for_target = stats;
                        }
                        break;
                    }
                }
                
                if (!found_next) {
                    break; // No se pudo agregar la caja, se "saturó"
                } else {
                    current_boxes++;
                    if (current_boxes == target_box) {
                        reached_target = true;
                    }
                }
            }
            
            if (reached_target) {
                unsigned int rows = current_board.size();
                unsigned int cols = current_board[0].size();
                std::lock_guard<std::mutex> lock(file_mutex);
                
                // Solo guardar si aun no llenamos la cuota
                if (box_counts[target_box].load() < per_box_target && saved_count < target_size) {
                    StructuralFeatures top_metrics = StructuralMetricsCalculator::calculate(current_board);
                    std::string serialized = serialize_board(current_board);
                    
                    file << "\"" << serialized << "\"," 
                         << cols << "," << rows << "," << target_box << "," 
                         << 1 << "," << shell_hash << ","
                         << final_stats_for_target.runtime_ms << ","
                         << final_stats_for_target.pushes << ","
                         << final_stats_for_target.moves << ","
                         << final_stats_for_target.generated_states << ","
                         << final_stats_for_target.expanded_nodes << ","
                         << final_stats_for_target.total_children << ","
                         << final_stats_for_target.effective_children << ","
                         << final_stats_for_target.repeated_nodes << ","
                         << final_stats_for_target.deadlocks << ","
                         << final_stats_for_target.branching_real << ","
                         << final_stats_for_target.branching_effective << ","
                         << final_stats_for_target.branching_classic << ","
                         << final_stats_for_target.redundancy << ","
                         << final_stats_for_target.closed_list_length << ","
                         << final_stats_for_target.path_stats.get_branching_real_avg() << ","
                         << final_stats_for_target.path_stats.branching_real_min << ","
                         << final_stats_for_target.path_stats.branching_real_max << ","
                         << final_stats_for_target.path_stats.get_branching_effective_avg() << ","
                         << final_stats_for_target.path_stats.branching_effective_min << ","
                         << final_stats_for_target.path_stats.branching_effective_max << ","
                         << final_stats_for_target.path_stats.get_redundancy() << ","
                         << final_stats_for_target.path_stats.deadlocks << ","
                         << final_stats_for_target.path_stats.box_lines << ","
                         << final_stats_for_target.path_stats.box_changes << ",\""
                         << final_stats_for_target.lurd_path << "\","
                         << top_metrics.wall_density << ","
                         << top_metrics.open_space_ratio << ","
                         << top_metrics.connectivity << ","
                         << top_metrics.aspect_ratio << ","
                         << top_metrics.dead_end_ratio << ","
                         << top_metrics.avg_symmetry << ","
                         << top_metrics.num_interior_regions << ","
                         << final_stats_for_target.initial_optimal_distance << "\n" << std::flush;
                         
                    saved_count++;
                    box_counts[target_box]++;
                    
                    if (saved_count % 10 == 0) {
                        std::cout << "Progreso: " << saved_count << " / " << target_size 
                                  << " | Cajas (1:" << box_counts[1] << " 2:" << box_counts[2] 
                                  << " 3:" << box_counts[3] << " 4:" << box_counts[4] 
                                  << " 5:" << box_counts[5] << " 6:" << box_counts[6] << " )\n";
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
    for (int b = 1; b <= 6; b++) {
        std::cout << "Cajas " << b << ": " << box_counts[b] << "\n";
    }
    
    return 0;
}
