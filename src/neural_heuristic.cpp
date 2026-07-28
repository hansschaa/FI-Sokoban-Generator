#include "neural_heuristic.hpp"
#include "constant.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "evolution/utils/board_utils.h"

NeuralHeuristic::NeuralHeuristic(const std::string& model_path, int rows, int cols) 
    : m(rows), n(cols) {
    try {
        // Load the TorchScript model
        model = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
        use_gpu = torch::cuda::is_available();
        if (use_gpu) {
            model->to(torch::kCUDA);
        }
        model->eval();
        
        // Disable gradients for faster inference
        torch::NoGradGuard no_grad;

        // Deadlock mask is computed lazily on first evaluate() call,
        // once end_vec (goal positions) is available.
        // compute_deadlock_mask() is NOT called here.

        // Allocate flat vector for the 6-channel tensor
        input_tensor_data.resize(6 * 25 * 25, 0.0f);
        
        // Load normalization stats
        std::ifstream stats_file("surrogate_models/results/surrogate_stats.txt");
        if (stats_file.is_open()) {
            stats_file >> pushes_mean >> pushes_std;
            stats_file.close();
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

void NeuralHeuristic::compute_deadlock_mask(const std::vector<std::vector<bool>>& end_vec) {
    // Build shell with walls AND goals — goals must never be flagged as deadlocks
    // (a box pushed onto a goal is a win condition, not a deadlock)
    std::vector<std::vector<char>> shell(m, std::vector<char>(n, ' '));
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            if (constant::blank_matrix[r][c] == constant::WALL) {
                shell[r][c] = '#';
            } else if (r < (int)end_vec.size() && c < (int)end_vec[r].size() && end_vec[r][c]) {
                shell[r][c] = '.'; // Mark goal — prevents it from being a deadlock cell
            }
        }
    }
    deadlock_mask = ::compute_deadlock_mask(shell);
    mask_initialized = true;
}

NeuralHeuristic::~NeuralHeuristic() {}

float NeuralHeuristic::evaluate(const game_node* node, const std::vector<std::vector<bool>>& end_vec) {
    // Lazy-init deadlock mask with goal positions on first call
    if (!mask_initialized) {
        compute_deadlock_mask(end_vec);
    }

    // Zero out the input tensor
    std::fill(input_tensor_data.begin(), input_tensor_data.end(), 0.0f);

    int max_h = 25;
    int max_w = 25;

    // Centering offsets
    int offset_r = 0; // (max_h - m) / 2; -- Removed centering to match Python training pipeline
    int offset_c = 0; // (max_w - n) / 2; -- Removed centering to match Python training pipeline

    // Channel 0: Walls
    // Channel 1: Floor / Blanks
    // Channel 2: Boxes
    // Channel 3: Targets
    // Channel 4: Player
    // Channel 5: Deadlock Mask

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

    // 5. Deadlock Mask (Channel 5)
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            if (deadlock_mask[r][c]) {
                int rr = r + offset_r;
                int cc = c + offset_c;
                input_tensor_data[5 * max_h * max_w + rr * max_w + cc] = 1.0f;
            }
        }
    }

    // Create tensor from data pointer
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(use_gpu ? torch::kCUDA : torch::kCPU);
    torch::Tensor input_tensor = torch::from_blob(input_tensor_data.data(), {1, 6, 25, 25}, torch::TensorOptions().dtype(torch::kFloat32));
    if (use_gpu) {
        input_tensor = input_tensor.to(torch::kCUDA);
    }

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

    // Lazy-init deadlock mask with goal positions on first call
    if (!mask_initialized) {
        compute_deadlock_mask(end_vec);
    }

    int N = nodes.size();
    int max_h = 25;
    int max_w = 25;
    int offset_r = 0; // (max_h - m) / 2; -- Removed centering to match Python training pipeline
    int offset_c = 0; // (max_w - n) / 2; -- Removed centering to match Python training pipeline

    std::vector<float> batch_data(N * 6 * max_h * max_w, 0.0f);
    
    // Default background is Wall (Channel 0 = 1.0) for all nodes
    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < max_h; ++r) {
            for (int c = 0; c < max_w; ++c) {
                batch_data[i * 6 * max_h * max_w + 0 * max_h * max_w + r * max_w + c] = 1.0f;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        const game_node* node = nodes[i];
        int b_idx = i * 6 * max_h * max_w;

        // 1. Static Layout
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                int rr = r + offset_r;
                int cc = c + offset_c;
                
                batch_data[b_idx + 0 * max_h * max_w + rr * max_w + cc] = 0.0f;

                if (constant::blank_matrix[r][c] == constant::WALL) {
                    batch_data[b_idx + 0 * max_h * max_w + rr * max_w + cc] = 1.0f;
                } else {
                    batch_data[b_idx + 1 * max_h * max_w + rr * max_w + cc] = 1.0f;
                }
            }
        }

        // 2. Targets
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (end_vec[r][c]) {
                    int rr = r + offset_r;
                    int cc = c + offset_c;
                    batch_data[b_idx + 3 * max_h * max_w + rr * max_w + cc] = 1.0f;
                }
            }
        }

        // 3. Boxes
        for (int j = 0; j < node->box_count; ++j) {
            int rr = node->box_list[j].x + offset_r;
            int cc = node->box_list[j].y + offset_c;
            batch_data[b_idx + 2 * max_h * max_w + rr * max_w + cc] = 1.0f;
        }

        // 4. Player
        int pr = node->person_point.x + offset_r;
        int pc = node->person_point.y + offset_c;
        batch_data[b_idx + 4 * max_h * max_w + pr * max_w + pc] = 1.0f;
        
        // 5. Deadlock Mask
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (deadlock_mask[r][c]) {
                    int rr = r + offset_r;
                    int cc = c + offset_c;
                    batch_data[b_idx + 5 * max_h * max_w + rr * max_w + cc] = 1.0f;
                }
            }
        }
    }

    auto input_tensor = torch::from_blob(batch_data.data(), {N, 6, max_h, max_w}, torch::kFloat32);
    if (use_gpu) {
        input_tensor = input_tensor.to(torch::kCUDA);
    }

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
