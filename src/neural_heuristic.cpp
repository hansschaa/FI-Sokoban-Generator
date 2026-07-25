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
        
        // Load normalization stats
        std::ifstream stats_file("surrogate_models/results/surrogate_stats.txt");
        if (stats_file.is_open()) {
            stats_file >> pushes_mean >> pushes_std;
            stats_file.close();
            // std::cout << "[NeuralHeuristic] Loaded stats: mean=" << pushes_mean << ", std=" << pushes_std << "\n";
        } else {
            std::cerr << "[NeuralHeuristic] Warning: surrogate_stats.txt not found. Using default raw stats (May cause severe inaccuracies!)\n";
            pushes_mean = 42.447f;
            pushes_std = 27.029f;
        }
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
    auto output = model->forward(inputs);
    
    // Regressor
    torch::Tensor pushes_pred;
    if (output.isTuple()) {
        pushes_pred = output.toTuple()->elements()[0].toTensor();
    } else {
        pushes_pred = output.toTensor();
    }
    
    float z_score = pushes_pred.item<float>();

    // Un-normalize
    // Apply expm1 to reverse the log1p normalization done in training
    float pushes_pred_val = std::expm1(z_score * pushes_std + pushes_mean);
    return std::max(0.0f, pushes_pred_val);
}

std::vector<float> NeuralHeuristic::evaluate_batch(const std::vector<const game_node*>& nodes, const std::vector<std::vector<bool>>& end_vec) {
    if (nodes.empty()) return {};

    int N = nodes.size();
    int max_h = 25;
    int max_w = 25;
    int offset_r = (max_h - m) / 2;
    int offset_c = (max_w - n) / 2;

    std::vector<float> batch_data(N * 5 * max_h * max_w, 0.0f);

    for (int i = 0; i < N; ++i) {
        const game_node* node = nodes[i];
        int batch_offset = i * 5 * max_h * max_w;

        // Default background is Wall (Channel 0 = 1.0)
        for (int r = 0; r < max_h; ++r) {
            for (int c = 0; c < max_w; ++c) {
                batch_data[batch_offset + 0 * max_h * max_w + r * max_w + c] = 1.0f;
            }
        }

        // 1. Static Layout (Walls and Floors)
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                int rr = r + offset_r;
                int cc = c + offset_c;
                int base_idx = batch_offset + rr * max_w + cc;
                
                batch_data[base_idx] = 0.0f; // Clear wall

                if (constant::blank_matrix[r][c] == constant::WALL) {
                    batch_data[base_idx] = 1.0f;
                } else {
                    batch_data[batch_offset + 1 * max_h * max_w + rr * max_w + cc] = 1.0f;
                }

                if (end_vec[r][c]) {
                    batch_data[batch_offset + 3 * max_h * max_w + rr * max_w + cc] = 1.0f;
                }
            }
        }

        // 2. Dynamic State (Boxes and Player)
        for (int b = 0; b < node->box_count; ++b) {
            int br = node->box_list[b].x + offset_r;
            int bc = node->box_list[b].y + offset_c;
            batch_data[batch_offset + 2 * max_h * max_w + br * max_w + bc] = 1.0f;
        }

        int pr = node->person_point.x + offset_r;
        int pc = node->person_point.y + offset_c;
        batch_data[batch_offset + 4 * max_h * max_w + pr * max_w + pc] = 1.0f;
    }

    auto input_tensor = torch::from_blob(batch_data.data(), {N, 5, max_h, max_w}, torch::kFloat32);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    torch::NoGradGuard no_grad;
    auto output = model->forward(inputs);
    
    torch::Tensor pushes_tensor;
    if (output.isTuple()) {
        pushes_tensor = output.toTuple()->elements()[0].toTensor();
    } else {
        pushes_tensor = output.toTensor();
    }

    std::vector<float> results;
    results.reserve(N);

    for (int i = 0; i < N; ++i) {
        float z_score = pushes_tensor[i].item<float>();
        // Apply expm1 to reverse the log1p normalization done in training
        float pushes_pred = std::expm1(z_score * pushes_std + pushes_mean);
        results.push_back(std::max(0.0f, pushes_pred));
    }

    return results;
}
