#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <mutex>
#include <future>

#include "shell_generator/shell_generator.h"
#include "game_solver.h"
#include "evolution/algorithms/evolution_strategy.h"
#include "evolution/evaluator/evaluator.h"
#include "evolution/utils/board_utils.h"
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
    
    int runs = 2000;
    std::string output_file = "es_dataset.csv";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--runs" && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    std::cout << "Starting ES Dataset Generation...\n";
    std::cout << "Target runs (base boards): " << runs << "\n";
    std::cout << "Output file: " << output_file << "\n\n";
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error opening output file!\n";
        return 1;
    }
    
    // Header
    file << "run_id,generation,pushes,board_string,width,height,boxes\n";
    std::mutex file_mutex;

    // Evaluator con FO1
    Evaluator evaluator;
    evaluator.fitnessType = FitnessType::FO1_PUSHES;
    
    auto global_start_time = std::chrono::high_resolution_clock::now();
    
    int successful_runs = 0;

    for (int run_id = 0; run_id < runs; run_id++) {
        std::cout << "\n=========================================\n";
        std::cout << "  RUN " << (run_id + 1) << " / " << runs << "\n";
        std::cout << "=========================================\n";

        int factorX = 2 + (rand() % 3); // 2, 3 o 4
        int factorY = 2 + (rand() % 3); // 2, 3 o 4

        SokobanGenerator generator(factorX, factorY);
        generator.generate();
        std::vector<std::vector<char>> shell = generator.getBoard();

        const int free_cells = count_free_cells(shell);
        if (free_cells < 5) {
            std::cout << "Shell too small, skipping...\n";
            continue;
        }

        auto deadlock_mask = compute_deadlock_mask(shell);
        
        // Inicializar poblacion
        std::vector<Individual> population;
        const int POP_SIZE = 10;
        std::mutex pop_mutex;

        auto generate_individual = [&](int i) {
            bool valid = false;
            int attempts = 0;
            while (!valid && attempts < 10000) {
                auto board = shell;
                int max_boxes = std::min(6, free_cells / 15);
                if(max_boxes < 1) max_boxes = 1;
                int numBoxes = 1; // Start simple

                bool placed = true;
                {
                    std::lock_guard<std::mutex> lock(pop_mutex);
                    try {
                        placeRandom(board, '@', deadlock_mask);
                        for (int k = 0; k < numBoxes; k++) {
                            placeRandom(board, '$', deadlock_mask);
                            placeRandom(board, '.', deadlock_mask);
                        }
                    } catch (...) {
                        placed = false;
                    }
                }
                
                if(!placed) {
                    attempts++;
                    continue;
                }

                std::string level = board_to_string(board);
                unsigned int rows = board.size();
                unsigned int cols = board[0].size();
                
                game_solver solver(level, rows, cols, 128);
                std::vector<game_node> solution;

                auto stats = solver.test_template(Method::a_star, solution);
                if (stats.status == SolveStatus::SOLVED) {
                    Individual ind;
                    ind.board = board;
                    
                    std::lock_guard<std::mutex> lock(pop_mutex);
                    ind.fitness = evaluator.evaluate(ind);
                    population.push_back(ind);
                    valid = true;
                }
                attempts++;
            }
        };

        // Generar poblacion inicial
        std::vector<std::future<void>> init_futures;
        for (int i = 0; i < POP_SIZE; i++) {
            init_futures.push_back(std::async(std::launch::async, generate_individual, i));
        }
        for (auto& f : init_futures) {
            f.get();
        }

        if (population.empty()) {
            std::cout << "Failed to generate initial population. Skipping run...\n";
            continue;
        }

        EvolutionStrategy es;
        es.evaluator = evaluator;
        es.use_parallel = true;
        es.setDeadlockMask(deadlock_mask);
        es.mu = 6;
        es.lambda = 30;
        es.mutationRate = 0.9878;
        es.maxEvaluations = 1000;
        es.stagnationLimit = 50;
        es.circuitStartTime = std::chrono::high_resolution_clock::now();

        // Callback para guardar el mejor de cada generacion
        es.on_generation = [&](int gen, const Individual& best_ind) {
            std::string serialized = serialize_board(best_ind.board);
            unsigned int rows = best_ind.board.size();
            unsigned int cols = best_ind.board.empty() ? 0 : best_ind.board[0].size();
            int boxes = 0;
            for(const auto& r : best_ind.board) {
                for(char c : r) {
                    if(c == '$' || c == '*') boxes++;
                }
            }
            
            std::lock_guard<std::mutex> lock(file_mutex);
            // run_id,generation,pushes,board_string,width,height,boxes
            file << run_id << ","
                 << gen << ","
                 << best_ind.fitness << ",\""
                 << serialized << "\","
                 << cols << ","
                 << rows << ","
                 << boxes << "\n";
        };

        std::cout << "Running ES for run " << run_id << "...\n";
        es.run(population);
        successful_runs++;
    }

    file.close();
    
    auto global_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(global_end_time - global_start_time);
    
    std::cout << "\nES Dataset generation completed!\n";
    std::cout << "Time elapsed: " << duration.count() << " seconds.\n";
    std::cout << "Successful base board runs: " << successful_runs << " / " << runs << "\n";
    
    return 0;
}
