#pragma once

#include <vector>
#include <string>
#include <memory>

#include "game_node.h"

// Forward declaration para no ensuciar el header con torch
namespace torch {
namespace jit {
    struct Module;
}
}

class NeuralHeuristic {
private:
    std::shared_ptr<torch::jit::Module> model;
    int m;
    int n;

    // Vector temporal para evitar realocaciones
    std::vector<float> input_tensor_data;

public:
    NeuralHeuristic(const std::string& model_path, int rows, int cols);
    ~NeuralHeuristic();

    float evaluate(const game_node* node, const std::vector<std::vector<bool>>& end_vec);
    std::vector<float> evaluate_batch(const std::vector<const game_node*>& nodes, const std::vector<std::vector<bool>>& end_vec);
};
