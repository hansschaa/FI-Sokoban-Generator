#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <mutex>
#include <future>
#include <map>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <thread>
#include <sstream>
#include <iomanip>

#include "shell_generator/shell_generator.h"
#include "game_solver.h"
#include "path_simulator.h"
#include "evolution/evaluator/evaluator.h"
#include "evolution/utils/board_utils.h"
#include "evolution/individual.h"

#include "evolution/mutations/add_mutation.h"
#include "evolution/mutations/move_mutation.h"
#include "evolution/mutations/remove_mutation.h"

namespace fs = std::filesystem;

// =====================================================================
// SERIALIZACION
// =====================================================================
std::string serialize_board(const std::vector<std::vector<char>>& board) {
    std::string s;
    for (size_t i = 0; i < board.size(); i++) {
        for (char c : board[i]) s += c;
        if (i < board.size() - 1) s += "\n";
    }
    return s;
}

// Evaluacion completa con path simulator activado
// Solo se llama para tableros que se van a guardar (no cada mutacion)
SolverStats evaluate_full(const std::vector<std::vector<char>>& board) {
    std::string level = board_to_string(board);
    unsigned int rows = board.size();
    unsigned int cols = board[0].size();

    game_solver solver(level, rows, cols, 64);
    solver.enable_advanced_deadlocks = true;

    std::vector<game_node> solution;
    // calc_path_branching = true para obtener stats del LURD
    return solver.test_template(Method::a_star, Heuristic::hungarian, solution, true);
}

// Serializa todas las stats en una linea
std::string stats_to_string(const SolverStats& s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    oss << "pushes:" << s.pushes
        << " moves:" << s.moves
        << " runtime_ms:" << s.runtime_ms
        << " generated_states:" << s.generated_states
        << " expanded_nodes:" << s.expanded_nodes
        << " total_children:" << s.total_children
        << " effective_children:" << s.effective_children
        << " repeated_nodes:" << s.repeated_nodes
        << " deadlocks:" << s.deadlocks
        << " branching_real:" << s.branching_real
        << " branching_effective:" << s.branching_effective
        << " branching_classic:" << s.branching_classic
        << " redundancy:" << s.redundancy
        << " closed_list:" << s.closed_list_length
        << " initial_dist:" << s.initial_optimal_distance;

    if (s.path_stats_calculated) {
        const auto& p = s.path_stats;
        oss << " path_states:" << p.states
            << " path_br_real_avg:" << p.get_branching_real_avg()
            << " path_br_eff_avg:" << p.get_branching_effective_avg()
            << " path_br_real_min:" << p.branching_real_min
            << " path_br_real_max:" << p.branching_real_max
            << " path_br_eff_min:" << p.branching_effective_min
            << " path_br_eff_max:" << p.branching_effective_max
            << " path_repeated:" << p.repeated_nodes
            << " path_deadlocks:" << p.deadlocks
            << " box_lines:" << p.box_lines
            << " box_changes:" << p.box_changes;
    }

    return oss.str();
}

// =====================================================================
// CLASE MINERO DE DATASET CON CUBETAS (v6: Stats Completas)
// =====================================================================
class SokobanMiner {
private:
    std::map<int, int> bucket_counts;
    std::map<int, std::unordered_set<int>> run_ids_in_bucket;
    const int BUCKET_CAPACITY = 1000;
    const int MAX_PUSHES_RANGE = 100;
    const int BUCKET_STEP = 10;
    fs::path base_dir;
    std::mutex miner_mutex;

    int getBucketId(int pushes) {
        if (pushes > MAX_PUSHES_RANGE) return MAX_PUSHES_RANGE + 1;
        return ((pushes - 1) / BUCKET_STEP) * BUCKET_STEP + 1;
    }

    std::string getBucketName(int bucket_id) {
        if (bucket_id > MAX_PUSHES_RANGE) return "101_plus";
        int upper_bound = bucket_id + BUCKET_STEP - 1;
        return std::to_string(bucket_id) + "_to_" + std::to_string(upper_bound);
    }

    // Carga los conteos de tableros existentes en los archivos .sok del directorio
    void loadExistingCounts() {
        if (!fs::exists(base_dir)) return;

        for (auto& entry : fs::directory_iterator(base_dir)) {
            if (entry.path().extension() != ".sok") continue;

            // Contar líneas con "pushes:" = número de tableros en ese archivo
            std::ifstream file(entry.path());
            int count = 0;
            std::string line;
            while (std::getline(file, line)) {
                if (line.find("pushes:") != std::string::npos) {
                    count++;
                    // Extraer el bucket_id del nombre del archivo (ej: "21_to_30.sok" → 21)
                }
            }

            if (count > 0) {
                // Parsear el bucket_id desde el nombre del archivo
                std::string stem = entry.path().stem().string(); // "21_to_30" o "101_plus"
                int bucket_id = -1;
                if (stem == "101_plus") {
                    bucket_id = MAX_PUSHES_RANGE + 1;
                } else {
                    // El nombre tiene forma "N_to_M", extraemos N
                    auto pos = stem.find("_to_");
                    if (pos != std::string::npos) {
                        bucket_id = std::stoi(stem.substr(0, pos));
                    }
                }

                if (bucket_id >= 0) {
                    bucket_counts[bucket_id] = count;
                    std::cout << "[Resume] Cubeta [" << stem << "]: " << count << "/" << BUCKET_CAPACITY << " tableros existentes." << std::endl;
                }
            }
        }
    }

public:
    SokobanMiner(const std::string& directory = "sokoban_dataset_buckets") : base_dir(directory) {
        if (!fs::exists(base_dir)) fs::create_directories(base_dir);
        // Cargar conteos existentes para reanudar sin sobreescribir
        loadExistingCounts();
        std::cout << "\n";
    }

    // Retorna true si el tablero debe ser guardado (pasa todos los filtros)
    // La evaluacion completa se hace FUERA del lock para no bloquear otros hilos
    bool shouldSave(int pushes, int run_id) {
        if (pushes <= 0) return false;
        int bucket = getBucketId(pushes);
        std::lock_guard<std::mutex> lock(miner_mutex);
        if (bucket_counts[bucket] >= BUCKET_CAPACITY) return false;
        if (run_ids_in_bucket[bucket].find(run_id) != run_ids_in_bucket[bucket].end()) return false;
        return true;
    }

    bool addBoard(const std::vector<std::vector<char>>& board, const SolverStats& stats, int run_id, std::mutex& cout_mutex) {
        int pushes = stats.pushes;
        if (pushes <= 0) return false;

        int bucket = getBucketId(pushes);

        std::lock_guard<std::mutex> lock(miner_mutex);
        if (bucket_counts[bucket] >= BUCKET_CAPACITY) return false;
        if (run_ids_in_bucket[bucket].find(run_id) != run_ids_in_bucket[bucket].end()) return false;

        std::string bucket_name = getBucketName(bucket);
        fs::path file_path = base_dir / (bucket_name + ".sok");

        std::ofstream outfile(file_path, std::ios::app);
        if (outfile.is_open()) {
            std::string board_str = serialize_board(board);
            size_t board_hash = std::hash<std::string>{}(board_str);

            outfile << board_hash << " - " << stats_to_string(stats) << "\n";
            outfile << board_str << "\n\n";
            outfile.close();

            bucket_counts[bucket]++;
            run_ids_in_bucket[bucket].insert(run_id);

            std::lock_guard<std::mutex> clog(cout_mutex);
            std::cout << "[Miner] Guardado (" << pushes << " empujes) en ["
                      << bucket_name << ".sok] -> (" << bucket_counts[bucket] << "/" << BUCKET_CAPACITY << ")\n";
            return true;
        }
        return false;
    }

    void printProgress(std::mutex& cout_mutex) {
        std::lock_guard<std::mutex> lock(miner_mutex);
        std::lock_guard<std::mutex> clog(cout_mutex);
        std::cout << "\n--- Estado de las Cubetas ---\n";
        for (const auto& [bucket, count] : bucket_counts) {
            std::cout << "Cubeta [" << getBucketName(bucket) << "]: " << count << "/" << BUCKET_CAPACITY << "\n";
        }
        std::cout << "-----------------------------\n";
    }
};

// FIX 1: Tamano uniforme 2-5 (sin size bias)
// FIX 2: Cajas iniciales aleatorias 1..min(5, espacio) (sin density bias)
bool generateBaseTemplate(std::vector<std::vector<char>>& board, std::vector<std::vector<bool>>& deadlock_mask) {
    int factorX = 2 + (rand() % 4);
    int factorY = 2 + (rand() % 4);

    SokobanGenerator generator(factorX, factorY);
    generator.generate();
    board = generator.getBoard();

    const int free_cells = count_free_cells(board);
    if (free_cells < 6) return false;

    deadlock_mask = compute_deadlock_mask(board);

    int max_initial = std::min(5, free_cells / 4);
    if (max_initial < 1) max_initial = 1;
    int numBoxes = 1 + (rand() % max_initial);

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
        if (arg == "--runs" && i + 1 < argc) runs = std::stoi(argv[++i]);
    }

    std::cout << "Starting Sokoban Dataset Miner (v6: Full Stats)...\n";
    std::cout << "Target base templates: " << runs << "\n\n";

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::cout << "Launching " << num_threads << " parallel miner threads...\n\n";

    SokobanMiner miner("sokoban_dataset_buckets");

    auto global_start_time = std::chrono::high_resolution_clock::now();
    std::atomic<int> successful_runs{0};
    std::atomic<int> current_run{0};
    std::mutex cout_mutex;

    auto worker_task = [&]() {
        // Evaluador rapido para guiar el ES (sin path simulator)
        Evaluator local_evaluator;
        local_evaluator.fitnessType = FitnessType::FO1_PUSHES;

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

            bool valid_base = false;
            double current_pushes = 0;
            Individual current_ind;

            int base_attempts = 0;
            while (!valid_base && base_attempts < 50) {
                base_attempts++;
                if (generateBaseTemplate(current_board, deadlock_mask)) {
                    current_ind.board = current_board;
                    current_pushes = local_evaluator.evaluate(current_ind);
                    if (current_pushes > 0 && !std::isnan(current_pushes)) {
                        valid_base = true;
                    }
                }
            }
            if (!valid_base) continue;

            MoveMutation moveMut;
            moveMut.deadlock_mask = deadlock_mask;
            AddMutation addMut;
            addMut.deadlock_mask = deadlock_mask;
            RemoveMutation removeMut;

            // Guardar snapshot base con stats completas (FIX 3)
            if (miner.shouldSave(static_cast<int>(current_pushes), run_id)) {
                SolverStats full_stats = evaluate_full(current_ind.board);
                if (full_stats.status == SolveStatus::SOLVED && full_stats.pushes > 0) {
                    miner.addBoard(current_ind.board, full_stats, run_id, cout_mutex);
                }
            }
            double last_snapshot_pushes = current_pushes;

            int failed_mutations = 0;
            const int MAX_PATIENCE = 3000;

            while (failed_mutations < MAX_PATIENCE) {
                Individual child = current_ind;
                bool success = false;

                int mutationType = rand() % 3;
                if (mutationType == 0)      success = moveMut.apply(child);
                else if (mutationType == 1) success = addMut.apply(child);
                else                        success = removeMut.apply(child);

                if (!success) { failed_mutations++; continue; }

                // Evaluacion rapida para guiar el ES
                double child_pushes = local_evaluator.evaluate(child);
                if (std::isnan(child_pushes) || child_pushes <= 0) { failed_mutations++; continue; }

                // FIX 3: Snapshot cada +10 empujes — evaluacion completa solo al guardar
                if (child_pushes >= last_snapshot_pushes + 10 &&
                    miner.shouldSave(static_cast<int>(child_pushes), run_id)) {
                    SolverStats full_stats = evaluate_full(child.board);
                    if (full_stats.status == SolveStatus::SOLVED && full_stats.pushes > 0) {
                        miner.addBoard(child.board, full_stats, run_id, cout_mutex);
                        last_snapshot_pushes = child_pushes;
                    }
                }

                if (child_pushes > current_pushes) {
                    current_ind = child;
                    current_pushes = child_pushes;
                    failed_mutations = 0;
                } else {
                    failed_mutations++;
                }
            }

            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[Anti-Estancamiento] (Run " << (run_id + 1) << ") Paciencia agotada ("
                          << MAX_PATIENCE << " intentos). Mejor: " << current_pushes << " empujes.\n";
            }

            miner.printProgress(cout_mutex);
            successful_runs++;
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; t++) threads.push_back(std::thread(worker_task));
    for (auto& t : threads) if (t.joinable()) t.join();

    auto global_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(global_end_time - global_start_time);

    std::cout << "\nMiner generation completed!\n";
    std::cout << "Time elapsed: " << duration.count() << " seconds.\n";
    std::cout << "Successful base board runs: " << successful_runs << " / " << runs << "\n";

    return 0;
}
