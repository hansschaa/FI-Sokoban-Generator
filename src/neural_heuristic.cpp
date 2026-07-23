#include "neural_heuristic.hpp"
#include "constant.h"

#include <torch/script.h>
#include <iostream>

NeuralHeuristic::NeuralHeuristic(const std::string& model_path, int rows, int cols) 
    : m(rows), n(cols) {
    try {
        // Load the TorchScript model
        model = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
        model->eval();
        
        // Disable gradients for faster inference
        torch::NoGradGuard no_grad;

        // Allocate flat vector for the 5-channel tensor
        input_tensor_data.resize(5 * 25 * 25, 0.0f);
    }
    catch (const c10::Error& e) {
        std::cerr << "[NeuralHeuristic] Error loading the model: " << model_path << "\n";
        std::cerr << e.what() << "\n";
        exit(1);
    }
}

NeuralHeuristic::~NeuralHeuristic() {}

float NeuralHeuristic::evaluate(const game_node* node, const std::vector<std::vector<bool>>& end_vec) {
    // Zero out the input tensor
    std::fill(input_tensor_data.begin(), input_tensor_data.end(), 0.0f);

    int max_h = 25;
    int max_w = 25;

    // Centering offsets
    int offset_r = (max_h - m) / 2;
    int offset_c = (max_w - n) / 2;

    // Channel 0: Walls
    // Channel 1: Floor / Blanks
    // Channel 2: Boxes
    // Channel 3: Targets
    // Channel 4: Player

    // Default background is Wall (Channel 0 = 1.0) for the whole 25x25 grid
    for (int r = 0; r < max_h; ++r) {
        for (int c = 0; c < max_w; ++c) {
            input_tensor_data[0 * max_h * max_w + r * max_w + c] = 1.0f;
        }
    }

    // 1. Static Layout (Walls and Floors)
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            int rr = r + offset_r;
            int cc = c + offset_c;
            int base_idx = rr * max_w + cc;
            
            // Clear the background wall for the actual map area
            input_tensor_data[0 * max_h * max_w + base_idx] = 0.0f;

            if (constant::blank_matrix[r][c] == constant::WALL) {
                input_tensor_data[0 * max_h * max_w + base_idx] = 1.0f;
            } else {
                input_tensor_data[1 * max_h * max_w + base_idx] = 1.0f;
            }
        }
    }

    // 2. Targets (Channel 3)
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            if (end_vec[r][c]) {
                int rr = r + offset_r;
                int cc = c + offset_c;
                input_tensor_data[3 * max_h * max_w + rr * max_w + cc] = 1.0f;
            }
        }
    }

    // 3. Boxes (Channel 2)
    for (int i = 0; i < node->box_count; ++i) {
        int rr = node->box_list[i].x + offset_r;
        int cc = node->box_list[i].y + offset_c;
        input_tensor_data[2 * max_h * max_w + rr * max_w + cc] = 1.0f;
    }

    // 4. Player (Channel 4)
    int pr = node->person_point.x + offset_r;
    int pc = node->person_point.y + offset_c;
    input_tensor_data[4 * max_h * max_w + pr * max_w + pc] = 1.0f;

    // Create tensor from data pointer
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor input_tensor = torch::from_blob(input_tensor_data.data(), {1, 5, 25, 25}, options);

    // Inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    torch::NoGradGuard no_grad;
    auto output = model->forward(inputs).toTuple();
    
    // Regressor returns: pushes_pred (B,), branching_pred (B,)
    torch::Tensor pushes_pred = output->elements()[0].toTensor();
    float z_score = pushes_pred.item<float>();

    // Un-normalize
    float pushes_mean = 42.44725765719096f;
    float pushes_std = 27.029350555210492f;
    
    float predicted_pushes = (z_score * pushes_std) + pushes_mean;

    return std::max(0.0f, predicted_pushes);
}
