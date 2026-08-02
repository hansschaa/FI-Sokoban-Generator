#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <random>
#include <chrono>

#include "game_solver.h"
#include "path_simulator.h"
#include "evolution/evaluator/evaluator.h"
#include "evolution/utils/board_utils.h"
#include "evolution/individual.h"

#include "evolution/mutations/add_mutation.h"
#include "evolution/mutations/move_mutation.h"
#include "evolution/mutations/remove_mutation.h"

namespace fs = std::filesystem;

std::vector<std::vector<char>> string_to_board(const std::string& str) {
    std::vector<std::vector<char>> temp_board;
    std::vector<char> row;
    size_t max_cols = 0;
    for (char c : str) {
        if (c == '\n') {
            if (!row.empty()) {
                if (row.size() > max_cols) max_cols = row.size();
                temp_board.push_back(row);
                row.clear();
            }
        } else {
            row.push_back(c);
        }
    }
    if (!row.empty()) {
        if (row.size() > max_cols) max_cols = row.size();
        temp_board.push_back(row);
    }
    
    // Pad rows
    for (auto& r : temp_board) {
        while (r.size() < max_cols) r.push_back(' ');
    }
    return temp_board;
}

std::string board_to_string_with_newlines(const std::vector<std::vector<char>>& board) {
    std::string s;
    for (size_t i = 0; i < board.size(); i++) {
        for (char c : board[i]) s += c;
        if (i < board.size() - 1) s += "\n";
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./solvable_pair_miner <solvables_dir> <output_dir> <num_pairs>\n";
        return 1;
    }

    std::string solvables_dir = argv[1];
    std::string output_dir = argv[2];
    int num_pairs = std::stoi(argv[3]);

    fs::create_directories(output_dir);

    std::vector<std::string> base_boards;
    for (const auto& entry : fs::recursive_directory_iterator(solvables_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".sok") {
            std::ifstream file(entry.path());
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            
            size_t pos = 0;
            while (pos < content.length()) {
                size_t end_pos = content.find("\n\n", pos);
                if (end_pos == std::string::npos) end_pos = content.length();
                std::string block = content.substr(pos, end_pos - pos);
                
                std::string cleaned;
                size_t bpos = 0;
                while (bpos < block.length()) {
                    size_t nl = block.find('\n', bpos);
                    if (nl == std::string::npos) nl = block.length();
                    std::string line = block.substr(bpos, nl - bpos);
                    if (line.find("Test -") == std::string::npos && 
                        line.find("type:") == std::string::npos && 
                        line.find("pushes:") == std::string::npos && 
                        line.find("runtime_ms:") == std::string::npos &&
                        line.find("label:") == std::string::npos &&
                        line.find("source_board:") == std::string::npos &&
                        line.find("mutated_board:") == std::string::npos) {
                        if (!line.empty() && line.find('#') != std::string::npos) {
                            cleaned += line + "\n";
                        }
                    }
                    bpos = nl + 1;
                }
                while (!cleaned.empty() && cleaned.back() == '\n') cleaned.pop_back();

                if (!cleaned.empty()) {
                    base_boards.push_back(cleaned);
                }
                pos = end_pos + 2;
            }
        }
    }

    std::cout << "Loaded " << base_boards.size() << " solvable boards.\n";
    if (base_boards.empty()) return 1;

    AddMutation addMutation;
    MoveMutation moveMutation;
    RemoveMutation removeMutation;

    Evaluator evaluator;
    evaluator.heuristic_type = Heuristic::hungarian;
    evaluator.use_surrogate = false;
    evaluator.fitnessType = FitnessType::FO1_PUSHES;

    srand(time(NULL));

    int generated = 0;
    
    std::ofstream out_file(output_dir + "/ranknet_intra_shell_pairs.sok");
    std::cout << "Generating " << num_pairs << " Siamese RankNet distance-1 pairs verified with real A*...\n";

    while (generated < num_pairs) {
        int idx = rand() % base_boards.size();
        Individual ind_parent;
        ind_parent.board = string_to_board(base_boards[idx]);
        std::string parent_str = board_to_string_with_newlines(ind_parent.board);
        
        double parent_pushes = evaluator.evaluate(ind_parent);
        if (parent_pushes <= 0 || parent_pushes < -100) continue;

        Individual ind_mutated = ind_parent;
        int mutationType = rand() % 3;
        bool success = false;
        
        if (mutationType == 0) success = moveMutation.apply(ind_mutated);
        else if (mutationType == 1) success = addMutation.apply(ind_mutated);
        else success = removeMutation.apply(ind_mutated);

        if (!success) continue;
        std::string mutated_str = board_to_string_with_newlines(ind_mutated.board);
        if (mutated_str == parent_str) continue;

        double mutated_pushes = evaluator.evaluate(ind_mutated);
        
        if (mutated_pushes > 0 && mutated_pushes > -100) {
            out_file << "source_board:\n" << parent_str << "\n";
            out_file << "source_pushes:" << static_cast<int>(parent_pushes) << "\n";
            out_file << "mutated_board:\n" << mutated_str << "\n";
            out_file << "mutated_pushes:" << static_cast<int>(mutated_pushes) << "\n\n";
            generated++;
            if (generated % 500 == 0) {
                std::cout << "Generated " << generated << "/" << num_pairs << " pairs (latest: " 
                          << static_cast<int>(parent_pushes) << " vs " << static_cast<int>(mutated_pushes) << " pushes).\n";
            }
        }
    }

    out_file.close();
    std::cout << "Done generating " << num_pairs << " RankNet solvable pairs saved to " << output_dir << "/ranknet_intra_shell_pairs.sok\n";
    return 0;
}
