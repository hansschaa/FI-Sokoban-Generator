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
    void compute_dist_to_goal(const std::vector<std::vector<bool>>& end_vec);
    bool mask_initialized = false;
    bool dist_initialized = false;
    bool use_gpu = false;
    bool disable_hybrid_switch = false;
    
    std::vector<point> goal_positions;
    std::vector<std::vector<std::vector<int>>> dist_to_goal;


    // Buffers pre-asignados para evitar malloc en cada llamada (Fix 2)
    std::vector<float> input_tensor_data;       // para evaluate() secuencial [6*25*25]
    std::vector<float> batch_tensor_data;       // para evaluate_batch()      [MAX_BATCH*6*25*25]
    static constexpr int MAX_BATCH_SIZE = 256;  // max nodos por batch
    
    // Normalization stats
    float pushes_mean = 3.4583510149067713f;
    float pushes_std = 0.8746270035752796f;

    // Calibration
    bool has_calibration = false;
    std::vector<double> calib_X;
    std::vector<double> calib_y;
    double calib_X_min = 0.0;
    double calib_X_max = 0.0;
    void load_calibration();
    float apply_calibration(float raw_pred);

public:
    std::vector<float> get_last_tensor() const { return input_tensor_data; }
    NeuralHeuristic(const std::string& model_path, int rows, int cols);
    void reset_board(int rows, int cols);
    ~NeuralHeuristic();

    float evaluate(const game_node* node, const std::vector<std::vector<bool>>& end_vec);
    std::vector<float> evaluate_batch(const std::vector<const game_node*>& nodes, const std::vector<std::vector<bool>>& end_vec);
};
