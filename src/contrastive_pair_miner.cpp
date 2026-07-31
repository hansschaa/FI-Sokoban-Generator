#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <random>
#include <chrono>

#include "game_solver.h"
#include "evolution/utils/board_utils.h"
#include "evolution/individual.h"
#include "evolution/mutations/add_mutation.h"
#include "evolution/mutations/move_mutation.h"
#include "evolution/mutations/remove_mutation.h"
#include "locked.h"

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
    if (argc < 4) {
        std::cerr << "Usage: ./contrastive_pair_miner <solvables_dir> <output_dir> <num_pairs>\n";
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
                    if (line.find("Test -") == std::string::npos && line.find("type:") == std::string::npos && line.find("pushes:") == std::string::npos && line.find("label:") == std::string::npos && line.find("source_board:") == std::string::npos && line.find("mutated_board:") == std::string::npos) {
                        if (!line.empty()) cleaned += line + "\n";
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

    srand(time(NULL));

    int label_1_count = 0;
    int label_0_simple_count = 0;
    int label_0_complex_count = 0;
    int timeout_count = 0;
    int total_processed = 0;

    std::ofstream out_label_1(output_dir + "/label_1_solvables.sok");
    std::ofstream out_label_0(output_dir + "/label_0_deadlocks.sok");

    while ((label_1_count + label_0_simple_count + label_0_complex_count) < num_pairs) {
        int idx = rand() % base_boards.size();
        Individual ind;
        ind.board = string_to_board(base_boards[idx]);
        std::string parent_str = board_to_string_with_newlines(ind.board);
        
        int mutationType = rand() % 3;
        bool success = false;
        
        if (mutationType == 0) success = moveMutation.apply(ind);
        else if (mutationType == 1) success = addMutation.apply(ind);
        else success = removeMutation.apply(ind);

        if (!success) continue;
        
        std::string mutated_str = board_to_string_with_newlines(ind.board);
        if (parent_str == mutated_str) continue;

        total_processed++;

        unsigned int rows = ind.board.size();
        unsigned int cols = ind.board[0].size();
        std::string flat_str = board_to_string(ind.board);
        // We use a small memory limit (16MB) so it fails quickly if it's too complex or unsolvable
        // instead of taking minutes to timeout.
        game_solver solver(flat_str, rows, cols, 16); 
        solver.enable_advanced_deadlocks = true;

        bool simple_deadlock = false;
        for (size_t i = 0; i < ind.board.size(); i++) {
            for (size_t j = 0; j < ind.board[i].size(); j++) {
                if (ind.board[i][j] == '$' || ind.board[i][j] == '*') {
                    point p(i, j);
                    if (solver.lk.is_locked(p, ind.board) || solver.lk.is_freeze_deadlock(p, ind.board)) {
                        simple_deadlock = true;
                        break;
                    }
                }
            }
            if (simple_deadlock) break;
        }

        // Fast rule check
        if (simple_deadlock) {
            out_label_0 << "source_board:\n" << parent_str << "\n";
            out_label_0 << "mutated_board:\n" << mutated_str << "\n";
            out_label_0 << "label:0\ntype:simple\n\n";
            label_0_simple_count++;
        } else {
            // A* check
            std::vector<game_node> solution;
            
            try {
                auto stats = solver.test_template(Method::a_star, Heuristic::hungarian, solution, false);
                
                if (stats.status == SolveStatus::SOLVED) {
                    out_label_1 << "source_board:\n" << parent_str << "\n";
                    out_label_1 << "mutated_board:\n" << mutated_str << "\n";
                    out_label_1 << "label:1\npushes:" << stats.pushes << "\n\n";
                    label_1_count++;
                } else if (stats.status == SolveStatus::UNSOLVABLE) {
                    out_label_0 << "source_board:\n" << parent_str << "\n";
                    out_label_0 << "mutated_board:\n" << mutated_str << "\n";
                    out_label_0 << "label:0\ntype:complex\n\n";
                    label_0_complex_count++;
                } else if (stats.status == SolveStatus::TIMEOUT) {
                    timeout_count++;
                }
            } catch (const std::runtime_error& e) {
                // OOM counts as timeout/discard
                timeout_count++;
            }
        }
        
        if (total_processed % 50 == 0) {
            std::cout << "Generated: " << (label_1_count + label_0_simple_count + label_0_complex_count) 
                      << " | L1: " << label_1_count 
                      << " | L0(Simp): " << label_0_simple_count
                      << " | L0(Comp): " << label_0_complex_count
                      << " | Timeouts (or OOM): " << timeout_count << std::endl;
            out_label_0.flush();
            out_label_1.flush();
        }
    }

    out_label_1.close();
    out_label_0.close();

    std::cout << "\n--- FINAL STATS ---\n";
    std::cout << "Label 1 (Solvable): " << label_1_count << "\n";
    std::cout << "Label 0 Simple (Corner/Freeze): " << label_0_simple_count << "\n";
    std::cout << "Label 0 Complex (CORRAL/A* Unsolvable): " << label_0_complex_count << "\n";
    std::cout << "Discarded due to Timeout: " << timeout_count << "\n";
    std::cout << "Total processed: " << total_processed << "\n";

    return 0;
}
