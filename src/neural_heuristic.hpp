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
    std::vector<std::vector<bool>> deadlock_mask;

    void compute_deadlock_mask(const std::vector<std::vector<bool>>& end_vec);
    bool mask_initialized = false;
    bool use_gpu = false;

    // Buffers pre-asignados para evitar malloc en cada llamada (Fix 2)
    std::vector<float> input_tensor_data;       // para evaluate() secuencial [6*25*25]
    std::vector<float> batch_tensor_data;       // para evaluate_batch()      [MAX_BATCH*6*25*25]
    static constexpr int MAX_BATCH_SIZE = 256;  // max nodos por batch
    
    // Normalization stats
    float pushes_mean = 42.447f;
    float pushes_std = 27.029f;

public:
    std::vector<float> get_last_tensor() const { return input_tensor_data; }
    NeuralHeuristic(const std::string& model_path, int rows, int cols);
    ~NeuralHeuristic();

    float evaluate(const game_node* node, const std::vector<std::vector<bool>>& end_vec);
    std::vector<float> evaluate_batch(const std::vector<const game_node*>& nodes, const std::vector<std::vector<bool>>& end_vec);
};
