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
            
            // Assuming Solvables files just contain lines of boards separated by double newlines or something.
            // Let's split by double newline.
            size_t pos = 0;
            while (pos < content.length()) {
                size_t end_pos = content.find("\n\n", pos);
                if (end_pos == std::string::npos) end_pos = content.length();
                std::string block = content.substr(pos, end_pos - pos);
                
                // Strip lines that start with metadata if any
                std::string cleaned;
                size_t bpos = 0;
                while (bpos < block.length()) {
                    size_t nl = block.find('\n', bpos);
                    if (nl == std::string::npos) nl = block.length();
                    std::string line = block.substr(bpos, nl - bpos);
                    if (line.find("Test -") == std::string::npos && line.find("type:") == std::string::npos) {
                        cleaned += line + "\n";
                    }
                    bpos = nl + 1;
                }
                while (!cleaned.empty() && cleaned.back() == '\n') cleaned.pop_back();

                if (!cleaned.empty() && cleaned.find('#') != std::string::npos) {
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
    
    std::ofstream out_file(output_dir + "/solvable_pairs.sok");

    while (generated < num_pairs) {
        int idx = rand() % base_boards.size();
        Individual ind;
        ind.board = string_to_board(base_boards[idx]);
        std::string parent_str = base_boards[idx];
        
        int mutationType = rand() % 3;
        bool success = false;
        
        if (mutationType == 0) success = moveMutation.apply(ind);
        else if (mutationType == 1) success = addMutation.apply(ind);
        else success = removeMutation.apply(ind);

        if (!success) continue;

        double fitness = evaluator.evaluate(ind);
        
        if (fitness > -1e8) {
            // It's still solvable!
            out_file << "source_board:\n" << parent_str << "\n";
            out_file << "mutated_board:\n" << board_to_string_with_newlines(ind.board) << "\n";
            out_file << "pushes:" << fitness << "\n\n";
            generated++;
            if (generated % 1000 == 0) std::cout << "Generated " << generated << " pairs.\n";
        }
    }

    out_file.close();
    std::cout << "Done generating " << num_pairs << " solvable pairs.\n";
    return 0;
}
