#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "../include/game_solver.h"

std::string load_board_str(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) return "";
    std::string board, line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        board += line + "\n";
    }
    return board;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string board_str = load_board_str(argv[1]);
    
    // Contamos num_goals
    int num_goals = 0;
    for (char c : board_str) if (c == '.' || c == '*' || c == '+') num_goals++;
    
    int mm = 0, nn = 0;
    int current_cols = 0;
    for (char c : board_str) {
        if (c == '\n') {
            mm++;
            if (current_cols > nn) nn = current_cols;
            current_cols = 0;
        } else {
            current_cols++;
        }
    }
    
    game_solver solver(board_str, mm, nn, 100); 
    // int child_count = solver.count_init_children();
    int child_count = 0;
    
    std::cout << "CHILDREN_COUNT: " << child_count << std::endl;
    return 0;
}
