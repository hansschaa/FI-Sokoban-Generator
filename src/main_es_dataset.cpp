#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <mutex>
#include <future>
#include <map>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <thread>

#include "shell_generator/shell_generator.h"
#include "game_solver.h"
#include "evolution/evaluator/evaluator.h"
#include "evolution/utils/board_utils.h"
#include "evolution/individual.h"

#include "evolution/mutations/add_mutation.h"
#include "evolution/mutations/move_mutation.h"
#include "evolution/mutations/remove_mutation.h"

namespace fs = std::filesystem;

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

// =====================================================================
// CLASE MINERO DE DATASET CON CUBETAS
// =====================================================================
class SokobanMiner {
private:
    std::map<int, int> bucket_counts;
    std::map<int, std::unordered_set<int>> run_ids_in_bucket; // Rastrear qué bases ya contribuyeron a cada cubeta
    const int BUCKET_CAPACITY = 1000;
    const int MAX_PUSHES_RANGE = 100; // Para la cubeta de 101+
    const int BUCKET_STEP = 10;
    fs::path base_dir;
    std::mutex miner_mutex;

    int getBucketId(int pushes) {
        if (pushes > MAX_PUSHES_RANGE) return MAX_PUSHES_RANGE + 1; // 101+
        return ((pushes - 1) / BUCKET_STEP) * BUCKET_STEP + 1; 
    }

    std::string getBucketName(int bucket_id) {
        if (bucket_id > MAX_PUSHES_RANGE) {
            return "101_plus";
        }
        int upper_bound = bucket_id + BUCKET_STEP - 1;
        return std::to_string(bucket_id) + "_to_" + std::to_string(upper_bound);
    }

public:
    SokobanMiner(const std::string& directory = "sokoban_dataset_buckets") : base_dir(directory) {
        if (!fs::exists(base_dir)) {
            fs::create_directories(base_dir);
        }
    }

    bool addBoard(const std::vector<std::vector<char>>& board, int pushes, int run_id) {
        if (pushes <= 60) return false; // Ignoramos todos los tableros menores a 61 empujes

        int bucket = getBucketId(pushes);
        
        std::lock_guard<std::mutex> lock(miner_mutex);
        if (bucket_counts[bucket] >= BUCKET_CAPACITY) {
            return false;
        }

        // REGLA ESTRICTA: Solo 1 tablero por base (run_id) por cubeta
        if (run_ids_in_bucket[bucket].find(run_id) != run_ids_in_bucket[bucket].end()) {
            return false;
        }

        std::string bucket_name = getBucketName(bucket);
        fs::path file_path = base_dir / (bucket_name + ".sok");

        std::ofstream outfile(file_path, std::ios::app);
        if (outfile.is_open()) {
            std::string board_str = serialize_board(board);
            size_t board_hash = std::hash<std::string>{}(board_str);
            
            outfile << board_hash << " - " << pushes << "\n";
            outfile << board_str << "\n\n";
            outfile.close();
            
            bucket_counts[bucket]++;
            run_ids_in_bucket[bucket].insert(run_id); // Marcar esta base como ya usada en esta cubeta
            
            std::cout << "[Miner] Guardado tablero (" << pushes << " empujes) en [" 
                      << bucket_name << ".sok] -> (" << bucket_counts[bucket] << "/" << BUCKET_CAPACITY << ")\n";
            return true;
        }
        
        return false;
    }

    void printProgress() {
        std::lock_guard<std::mutex> lock(miner_mutex);
        std::cout << "\n--- Estado de las Cubetas ---\n";
        for (const auto& [bucket, count] : bucket_counts) {
            std::cout << "Cubeta [" << getBucketName(bucket) << "]: " << count << "/" << BUCKET_CAPACITY << "\n";
        }
        std::cout << "-----------------------------\n";
    }

    void getDynamicFactors(int& factorX, int& factorY) {
        std::lock_guard<std::mutex> lock(miner_mutex);
        
        // Forzamos Fase 3 permanentemente para buscar > 60 empujes
        factorX = 4 + (rand() % 2); // 4 o 5
        factorY = 4 + (rand() % 2); // 4 o 5
    }
};

// Generador de template base (cascaron)
bool generateBaseTemplate(std::vector<std::vector<char>>& board, std::vector<std::vector<bool>>& deadlock_mask, SokobanMiner& miner) {
    int factorX, factorY;
    miner.getDynamicFactors(factorX, factorY);

    SokobanGenerator generator(factorX, factorY);
    generator.generate();
    board = generator.getBoard();

    const int free_cells = count_free_cells(board);
    if (free_cells < 5) {
        return false;
    }

    deadlock_mask = compute_deadlock_mask(board);

    // Intentar colocar al menos un jugador y una caja
    int max_boxes = std::min(6, free_cells / 15);
    if(max_boxes < 1) max_boxes = 1;
    int numBoxes = 1;

    try {
        placeRandom(board, '@', deadlock_mask);
        for (int k = 0; k < numBoxes; k++) {
            placeRandom(board, '$', deadlock_mask);
            placeRandom(board, '.', deadlock_mask);
        }
    } catch (...) {
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    int runs = 2000;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "Starting Sokoban Dataset Miner...\n";
    std::cout << "Target base templates: " << runs << "\n\n";
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::cout << "Launching " << num_threads << " parallel miner threads...\n\n";

    SokobanMiner miner("sokoban_dataset_buckets");
    Evaluator evaluator;
    evaluator.fitnessType = FitnessType::FO1_PUSHES;
    
    auto global_start_time = std::chrono::high_resolution_clock::now();
    std::atomic<int> successful_runs{0};
    std::atomic<int> current_run{0};
    std::mutex cout_mutex;

    auto worker_task = [&]() {
        while (true) {
            int run_id = current_run.fetch_add(1);
            if (run_id >= runs) break;

            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "\n=========================================\n";
                std::cout << "  RUN " << (run_id + 1) << " / " << runs << "\n";
                std::cout << "=========================================\n";
            }

        std::vector<std::vector<char>> current_board;
        std::vector<std::vector<bool>> deadlock_mask;

        // 1. Reinicio: Generar un template base válido
        bool valid_base = false;
        double current_pushes = 0;
        Individual current_ind;

        while (!valid_base) {
            if (generateBaseTemplate(current_board, deadlock_mask, miner)) {
                current_ind.board = current_board;
                current_pushes = evaluator.evaluate(current_ind);
                if (current_pushes > 0 && !std::isnan(current_pushes)) {
                    valid_base = true;
                }
            }
        }

        // Instanciar mutaciones para este template
        MoveMutation moveMut;
        moveMut.deadlock_mask = deadlock_mask;
        AddMutation addMut;
        addMut.deadlock_mask = deadlock_mask;
        RemoveMutation removeMut;

        // Alimentar minero con base
        miner.addBoard(current_ind.board, static_cast<int>(current_pushes), run_id);

        int failed_mutations = 0;
        const int MAX_PATIENCE = 3000;

        // 2. Proceso Evolutivo (1+1)-ES
        while (failed_mutations < MAX_PATIENCE) {
            Individual child = current_ind;
            bool success = false;

            int mutationType = rand() % 3;
            if (mutationType == 0) {
                success = moveMut.apply(child);
            } else if (mutationType == 1) {
                success = addMut.apply(child);
            } else {
                success = removeMut.apply(child);
            }

            if (!success) {
                failed_mutations++;
                continue;
            }

            double child_pushes = evaluator.evaluate(child);

            // Descartar inmediatamente injugables
            if (std::isnan(child_pushes) || child_pushes <= 0) {
                failed_mutations++;
                continue;
            }

            // Alimentar minero
            miner.addBoard(child.board, static_cast<int>(child_pushes), run_id);

            // 3. Presión de Selección Estricta: MÁS DIFÍCIL
            if (child_pushes > current_pushes) {
                current_ind = child;
                current_pushes = child_pushes;
                failed_mutations = 0; // Reiniciamos paciencia por mejora
            } else {
                failed_mutations++;
            }
        }

            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[Anti-Estancamiento] (Run " << (run_id + 1) << ") Paciencia agotada (" << MAX_PATIENCE 
                          << " intentos). Mejor tablero alcanzó " << current_pushes << " empujes.\n";
            }
                  
            miner.printProgress();
            successful_runs++;
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; t++) {
        threads.push_back(std::thread(worker_task));
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto global_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(global_end_time - global_start_time);
    
    std::cout << "\nMiner generation completed!\n";
    std::cout << "Time elapsed: " << duration.count() << " seconds.\n";
    std::cout << "Successful base board runs: " << successful_runs << " / " << runs << "\n";
    
    return 0;
}
