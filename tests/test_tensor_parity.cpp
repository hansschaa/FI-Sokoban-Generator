#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <torch/torch.h>
#include "../include/game_solver.h"
#include "../src/neural_heuristic.hpp"
#include "../include/evolution/utils/board_utils.h"
#include "../include/constant.h"

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Uso: ./test_tensor_parity <board.txt>\n";
        return 1;
    }
    
    std::ifstream file(argv[1]);
    if (!file.is_open()) return 1;
    std::vector<std::vector<char>> board;
    std::string line;
    size_t n = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        n = std::max(n, line.length());
        board.emplace_back(line.begin(), line.end());
    }
    for (auto& row : board) while (row.size() < n) row.push_back(' ');
    int m = board.size();

    // Populate constant::blank_matrix
    constant::blank_matrix.assign(m, std::vector<char>(n, constant::BLANK));
    
    game_node node;
    node.box_count = 0;
    std::vector<std::vector<bool>> end_vec(m, std::vector<bool>(n, false));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < (int)n; ++j) {
            char c = board[i][j];
            if (c == '#') {
                constant::blank_matrix[i][j] = constant::WALL;
            } else if (c == '$' || c == '*') {
                node.box_list[node.box_count++] = point(i, j);
            } else if (c == '@' || c == '+') {
                node.person_point = point(i, j);
            }
            if (c == '.' || c == '*' || c == '+') {
                end_vec[i][j] = true;
            }
        }
    }
    
    // Create NeuralHeuristic
    NeuralHeuristic nh("surrogate_models/results/surrogate_regressor_jit.pt", m, n);
    nh.evaluate(&node, end_vec);
    
    std::vector<float> tensor = nh.get_last_tensor();
    
    std::cout << "--- BOARD 0 TENSORS ---\n";
    for (int ch = 0; ch < 6; ++ch) {
        std::cout << "Channel " << ch << ":\n";
        int offset = ch * 25 * 25;
        for (int r = 0; r < 25; ++r) {
            for (int c = 0; c < 25; ++c) {
                float val = tensor[offset + r * 25 + c];
                std::cout << (val > 0.5f ? '1' : '0');
            }
            std::cout << "\n";
        }
    }
    
    return 0;
}
