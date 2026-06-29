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
#include "evolution/mutations/move_mutation.h"
#include "evolution/individual.h"

// Usar \n real para que sea compatible con copy-paste a JSoko
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
    
    int target_size = 1000;
    std::string output_file = "dataset.csv";
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
    
    std::cout << "Starting Dataset Generation...\n";
    std::cout << "Target size: " << target_size << "\n";
    std::cout << "Output file: " << output_file << "\n\n";
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    
    // Header
    file << "board_string,width,height,boxes,is_solvable,dataset_type,shell_hash\n";
    
    std::atomic<int> saved_count{0};
    std::atomic<int> solvable_count{0};
    std::atomic<int> easy_negative_count{0};
    std::atomic<int> hard_negative_count{0};
    std::mutex file_mutex;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Función para el hilo trabajador
    auto worker_task = [&]() {
        while (saved_count < target_size) {
            
            int factorX = 2 + (rand() % 3); // 2, 3 o 4
            int factorY = 2 + (rand() % 3); // 2, 3 o 4
            
            SokobanGenerator generator(factorX, factorY);
            generator.generate();
            std::vector<std::vector<char>> shell = generator.getBoard();
            
            auto deadlock_mask = compute_deadlock_mask(shell);
            int free_cells = count_free_cells(shell);
            
            if (free_cells < 5) continue;
            
            std::string shell_str = board_to_string(shell);
            size_t shell_hash = std::hash<std::string>{}(shell_str);
            
            // Elegir num de cajas elevando el limite a 8 y aumentando densidad (free_cells / 3)
            int max_boxes = std::min(8, free_cells / 3);
            if (max_boxes < 1) max_boxes = 1;
            
            // Para evitar tableros medianos vacíos, el mínimo escala con el máximo
            int min_boxes = std::max(1, max_boxes / 2);
            int numBoxes = min_boxes + (rand() % (max_boxes - min_boxes + 1));
            
            // Colocar entidades
            auto board = shell;
            bool placement_success = true;
            
            try {
                placeRandom(board, '@', deadlock_mask);
                for (int k = 0; k < numBoxes; k++) {
                    placeRandom(board, '$', deadlock_mask);
                    placeRandom(board, '.', deadlock_mask);
                }
            } catch (const std::runtime_error& e) {
                placement_success = false;
            }
            
            if (!placement_success) continue;
            
            unsigned int rows = board.size();
            unsigned int cols = board[0].size();
            std::string serialized = serialize_board(board);
            bool is_positive = false;
            
            {
                std::string level = board_to_string(board);
                
                game_solver solver(level, rows, cols, 32);
                std::vector<game_node> solution;
                
                auto stats = solver.test_template(Method::a_star, solution);
                if (stats.status == SolveStatus::TIMEOUT) continue;
                is_positive = (stats.status == SolveStatus::SOLVED);
            }
            
            // Si es injugable (Easy Negative), lo guardamos pero limitando la cantidad para balancear
            if (!is_positive) {
                std::lock_guard<std::mutex> lock(file_mutex);
                if (easy_negative_count <= solvable_count + 10 && saved_count < target_size) {
                    std::string serialized = serialize_board(board);
                    file << "\"" << serialized << "\"," 
                         << cols << "," << rows << "," << numBoxes << "," 
                         << 0 << ",\"easy_negative\"," << shell_hash << "\n";
                    saved_count++;
                    easy_negative_count++;
                }
                continue; // Buscamos otra semilla aleatoria o pasamos a otro cascaron
            }
            
            // SI ES POSITIVO, COMENZAMOS LA CAMINATA MUTACIONAL
            int local_extracted = 0;
            Individual current_seed;
            current_seed.board = board;
            
            // Primero guardamos la semilla original
            {
                std::string serialized = serialize_board(board);
                std::lock_guard<std::mutex> lock(file_mutex);
                if (saved_count < target_size) {
                    file << "\"" << serialized << "\"," 
                         << cols << "," << rows << "," << numBoxes << "," 
                         << 1 << ",\"positive\"," << shell_hash << "\n";
                    saved_count++;
                    solvable_count++;
                    local_extracted++;
                }
            }
            
            MoveMutation mut;
            mut.deadlock_mask = deadlock_mask;
            
            // Extraer hasta 5 positivos de este cascaron
            int max_mutations_tries = 50; 
            int tries = 0;
            
            while (local_extracted < 5 && tries < max_mutations_tries && saved_count < target_size) {
                tries++;
                Individual candidate = current_seed;
                
                // Aplicar mutacion (max 5 intentos)
                bool mutated = false;
                for (int m = 0; m < 5; m++) {
                    if (mut.apply(candidate)) {
                        mutated = true;
                        break;
                    }
                }
                
                if (!mutated) continue;
                
                std::string level_mut = board_to_string(candidate.board);
                SolveStatus mut_status = SolveStatus::TIMEOUT;
                {
                    game_solver solver_mut(level_mut, rows, cols, 32);
                    std::vector<game_node> solution_mut;
                    auto stats_mut = solver_mut.test_template(Method::a_star, solution_mut);
                    mut_status = stats_mut.status;
                }
                
                if (mut_status == SolveStatus::TIMEOUT) continue;
                
                std::string serialized_mut = serialize_board(candidate.board);
                
                if (mut_status == SolveStatus::SOLVED) {
                    // Nuevo positivo encontrado! Se vuelve la nueva semilla
                    current_seed = candidate;
                    
                    std::lock_guard<std::mutex> lock(file_mutex);
                    if (saved_count < target_size) {
                        file << "\"" << serialized_mut << "\"," 
                             << cols << "," << rows << "," << numBoxes << "," 
                             << 1 << ",\"positive\"," << shell_hash << "\n";
                        saved_count++;
                        solvable_count++;
                        local_extracted++;
                    }
                } else if (mut_status == SolveStatus::UNSOLVABLE) {
                    // Hard negative encontrado
                    std::lock_guard<std::mutex> lock(file_mutex);
                    if (saved_count < target_size) {
                        file << "\"" << serialized_mut << "\"," 
                             << cols << "," << rows << "," << numBoxes << "," 
                             << 0 << ",\"hard_negative\"," << shell_hash << "\n";
                        saved_count++;
                        hard_negative_count++;
                    }
                }
            }
            
            if (saved_count % 10 == 0) {
                std::lock_guard<std::mutex> lock(file_mutex);
                std::cout << "Progreso: " << saved_count << " / " << target_size 
                          << " (Jugables: " << solvable_count 
                          << " | Injugables Faciles: " << easy_negative_count 
                          << " | Injugables Dificiles: " << hard_negative_count << ")\n";
            }
        }
    };
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (custom_threads > 0) num_threads = custom_threads;
    
    std::cout << "Using " << num_threads << " threads.\n";
    
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
    std::cout << "Total Jugables: " << solvable_count << "\n";
    std::cout << "Total Injugables (Easy): " << easy_negative_count << "\n";
    std::cout << "Total Injugables (Hard): " << hard_negative_count << "\n";
    
    return 0;
}
