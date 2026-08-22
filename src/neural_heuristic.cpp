#include "neural_heuristic.hpp"
#include "constant.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "evolution/utils/board_utils.h"
#include "hungarian.h"
#include <deque>
#include <array>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
    static void reverse_push_bfs(
        const point& goal,
        const std::vector<std::vector<char>>& blank_matrix,
        int m, int n,
        std::vector<std::vector<int>>& dist
    ) {
        const int INF = 1000;
        std::vector<std::vector<std::array<int,4>>> dist_state(
            m, std::vector<std::array<int,4>>(n, {INF, INF, INF, INF}));

        struct State { point box; int dir_idx; };
        std::deque<State> q;

        for (int d = 0; d < 4; d++) {
            point player_needed = goal - constant::four_direction[d];
            if (player_needed.x < 0 || player_needed.x >= m ||
                player_needed.y < 0 || player_needed.y >= n) continue;
            if (blank_matrix[player_needed.x][player_needed.y] == constant::WALL) continue;

            if (dist_state[goal.x][goal.y][d] == INF) {
                dist_state[goal.x][goal.y][d] = 0;
                q.push_back({goal, d});
            }
        }

        while (!q.empty()) {
            auto [box, dir_idx] = q.front();
            q.pop_front();

            int cur_cost = dist_state[box.x][box.y][dir_idx];

            for (int push_d = 0; push_d < 4; push_d++) {
                point box_prev = box - constant::four_direction[push_d];
                if (box_prev.x < 0 || box_prev.x >= m ||
                    box_prev.y < 0 || box_prev.y >= n) continue;
                if (blank_matrix[box_prev.x][box_prev.y] == constant::WALL) continue;

                point player_prev = box_prev - constant::four_direction[push_d];
                if (player_prev.x < 0 || player_prev.x >= m ||
                    player_prev.y < 0 || player_prev.y >= n) continue;
                if (blank_matrix[player_prev.x][player_prev.y] == constant::WALL) continue;

                int new_cost = cur_cost + 1;
                if (new_cost < dist_state[box_prev.x][box_prev.y][push_d]) {
                    dist_state[box_prev.x][box_prev.y][push_d] = new_cost;
                    q.push_back({box_prev, push_d});
                }
            }
        }

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                int best = INF;
                for (int d = 0; d < 4; d++)
                    best = std::min(best, dist_state[i][j][d]);
                dist[i][j] = best;
            }
    }
}

void NeuralHeuristic::compute_dist_to_goal(const std::vector<std::vector<bool>>& end_vec) {
    goal_positions.clear();
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (i < (int)end_vec.size() && j < (int)end_vec[i].size() && end_vec[i][j])
                goal_positions.push_back(point(i, j));

    int num_goals = (int)goal_positions.size();
    dist_to_goal.assign(num_goals, std::vector<std::vector<int>>(m, std::vector<int>(n, 1000)));

    for (int g = 0; g < num_goals; g++) {
        reverse_push_bfs(goal_positions[g], constant::blank_matrix, m, n, dist_to_goal[g]);
    }
    dist_initialized = true;
}

NeuralHeuristic::NeuralHeuristic(const std::string& model_path, int rows, int cols) 
    : m(rows), n(cols) {
    try {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // Load the TorchScript model
        model = std::make_shared<torch::jit::Module>(torch::jit::load(model_path));
        use_gpu = false;
        if (use_gpu) {
            model->to(torch::kCUDA);
        }
        model->eval();
        
        // Disable gradients for faster inference
        torch::NoGradGuard no_grad;

        // Dummy forward pass for CUDA / cuDNN warmup
        torch::Tensor dummy_input = torch::zeros({1, 6, 25, 25}, torch::TensorOptions().dtype(torch::kFloat32));
        if (use_gpu) dummy_input = dummy_input.to(torch::kCUDA);
        std::vector<torch::jit::IValue> dummy_inputs;
        dummy_inputs.push_back(dummy_input);
        model->forward(dummy_inputs);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double warmup_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[CUDA WARMUP] Costo fijo de inicializacion: " << warmup_ms << " ms" << std::endl;

        // Deadlock mask is computed lazily on first evaluate() call,
        // once end_vec (goal positions) is available.
        // compute_deadlock_mask() is NOT called here.

        // Allocate flat vector for the 6-channel tensor (sequential)
        input_tensor_data.resize(6 * 25 * 25, 0.0f);
        
        // Pre-alloc batch buffer (Fix 2): evita malloc en cada llamada a evaluate_batch
        batch_tensor_data.resize(MAX_BATCH_SIZE * 6 * 25 * 25, 0.0f);
        
        // Load normalization stats
        std::ifstream stats_file("surrogate_models/results/surrogate_stats.txt");
        if (stats_file.is_open()) {
            stats_file >> pushes_mean >> pushes_std;
            stats_file.close();
        } else {
            std::cerr << "[NeuralHeuristic] Warning: surrogate_stats.txt not found. Using default raw stats (May cause severe inaccuracies!)\n";
            pushes_mean = 3.4583510149067713f;
            pushes_std = 0.8746270035752796f;
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
    // (a box pushed onto a goal is a win condition, not a deadlock)    // Load calibration if available
    load_calibration();

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

void NeuralHeuristic::load_calibration() {
    if (has_calibration) return;
    std::ifstream f("surrogate_models/results/regressor_calibration.json");
    if (!f.is_open()) {
        std::cerr << "[NeuralHeuristic] Warning: regressor_calibration.json not found. Running uncalibrated.\n";
        return;
    }
    try {
        json j;
        f >> j;
        calib_X = j["X_thresholds"].get<std::vector<double>>();
        calib_y = j["y_thresholds"].get<std::vector<double>>();
        calib_X_min = j["X_min"].get<double>();
        calib_X_max = j["X_max"].get<double>();
        has_calibration = true;
    } catch (const std::exception& e) {
        std::cerr << "[NeuralHeuristic] Error parsing calibration JSON: " << e.what() << "\n";
    }
}

float NeuralHeuristic::apply_calibration(float raw_pred) {
    if (!has_calibration || calib_X.empty()) return raw_pred;
    
    double x = (double)raw_pred;
    if (x <= calib_X_min) return (float)calib_y.front();
    if (x >= calib_X_max) return (float)calib_y.back();
    
    // Scikit-learn IsotonicRegression step interpolation
    auto it = std::upper_bound(calib_X.begin(), calib_X.end(), x);
    if (it == calib_X.end()) return (float)calib_y.back();
    if (it == calib_X.begin()) return (float)calib_y.front();
    
    size_t idx = std::distance(calib_X.begin(), it);
    // Linear interpolation between the two nearest thresholds
    double x0 = calib_X[idx - 1];
    double x1 = calib_X[idx];
    double y0 = calib_y[idx - 1];
    double y1 = calib_y[idx];
    
    double res = y0 + (x - x0) * (y1 - y0) / (x1 - x0);
    return (float)res;
}

float NeuralHeuristic::evaluate(const game_node* node, const std::vector<std::vector<bool>>& end_vec) {
    // Lazy-init deadlock mask with goal positions on first call
    if (!mask_initialized) {
        compute_deadlock_mask(end_vec);
    }
    if (!dist_initialized) {
        compute_dist_to_goal(end_vec);
    }

    // Hybrid Switch: Usar Hungarian puro para tableros con 6+ cajas
    // donde la red neuronal tiende a colapsar cognitivamente.
    if (node->box_count >= 6) {
        int num_boxes = node->box_count;
        int num_goals = (int)goal_positions.size();
        int sz = std::max(num_boxes, num_goals);
        std::vector<std::vector<int>> cost(sz, std::vector<int>(sz, 0));
        for (int b = 0; b < num_boxes; ++b) {
            for (int g = 0; g < num_goals; ++g) {
                cost[b][g] = dist_to_goal[g][node->box_list[b].x][node->box_list[b].y];
            }
        }
        Hungarian h(cost);
        return (float)h.solve();
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
    
    // Fix 1: mover tensor a CPU UNA sola vez, luego leer con accessor
    auto cpu_tensor = pushes_pred.to(torch::kCPU).contiguous();
    static std::ofstream dump_file("scratch/classifier_dump.txt", std::ios::app);
    dump_file << node->board_to_pretty_string() << "===" << "\n";
    auto accessor = cpu_tensor.accessor<float, 1>();
    static std::ofstream dump_file("scratch/classifier_dump.txt", std::ios::app);
    for (size_t i = 0; i < nodes.size(); i++) {
        dump_file << nodes[i]->board_to_pretty_string() << "===";
    }
    dump_file << "\n";
    dump_file.flush();
    float z_score = accessor[0];

    // Un-normalize
    // Apply expm1 to reverse the log1p normalization done in training
    float pushes_pred_val = std::expm1(z_score * pushes_std + pushes_mean);
    
    // Explicit numerical safety guard against NaN/Inf or astronomically exploded values
    if (!std::isfinite(pushes_pred_val) || pushes_pred_val > 1e6f || pushes_pred_val < 0.0f) {
        pushes_pred_val = 1e6f; // Fallback large value, will be clamped by calibration/admissibility anyway, or we can just cap it to avoid breaking calibration
    }
    
    // Apply Isotonic Calibration
    pushes_pred_val = apply_calibration(pushes_pred_val);
    
    if (dist_initialized && !goal_positions.empty()) {
        int num_boxes = node->box_count;
        int num_goals = (int)goal_positions.size();
        int sz = std::max(num_boxes, num_goals);
        std::vector<std::vector<int>> cost(sz, std::vector<int>(sz, 0));
        for (int b = 0; b < num_boxes; ++b) {
            for (int g = 0; g < num_goals; ++g) {
                cost[b][g] = dist_to_goal[g][node->box_list[b].x][node->box_list[b].y];
            }
        }
        Hungarian h(cost);
        float hungarian_lb = (float)h.solve();
        // Use the empirically derived 95th percentile K multiplier (1.318)
        pushes_pred_val = hungarian_lb + std::clamp(pushes_pred_val - hungarian_lb, 0.0f, 1.318f * hungarian_lb);
    }
    
    return pushes_pred_val;
}

std::vector<float> NeuralHeuristic::evaluate_batch(const std::vector<const game_node*>& nodes, const std::vector<std::vector<bool>>& end_vec) {
    if (nodes.empty()) return {};

    // Lazy-init deadlock mask with goal positions on first call
    if (!mask_initialized) {
        compute_deadlock_mask(end_vec);
    }
    if (!dist_initialized) {
        compute_dist_to_goal(end_vec);
    }

    // Hybrid Switch: Usar Hungarian puro para tableros con 6+ cajas
    if (nodes[0]->box_count >= 6) {
        std::vector<float> results;
        for (size_t i = 0; i < nodes.size(); ++i) {
            const game_node* node = nodes[i];
            
            // Check deadlock first (al igual que en modo neuronal)
            bool is_deadlock = false;
            for (int j = 0; j < node->box_count; ++j) {
                if (deadlock_mask[node->box_list[j].x][node->box_list[j].y]) {
                    is_deadlock = true;
                    break;
                }
            }
            if (is_deadlock) {
                results.push_back(10000.0f);
                continue;
            }

            int num_boxes = node->box_count;
            int num_goals = (int)goal_positions.size();
            int sz = std::max(num_boxes, num_goals);
            std::vector<std::vector<int>> cost(sz, std::vector<int>(sz, 0));
            for (int b = 0; b < num_boxes; ++b) {
                for (int g = 0; g < num_goals; ++g) {
                    cost[b][g] = dist_to_goal[g][node->box_list[b].x][node->box_list[b].y];
                }
            }
            Hungarian h(cost);
            results.push_back((float)h.solve());
        }
        return results;
    }

    int N = nodes.size();
    int max_h = 25;
    int max_w = 25;
    int offset_r = 0; // (max_h - m) / 2; -- Removed centering to match Python training pipeline
    int offset_c = 0; // (max_w - n) / 2; -- Removed centering to match Python training pipeline

    // Reutilizar el buffer pre-asignado (Fix 2): si el batch excede MAX_BATCH_SIZE, crecer dinámicamente
    int needed = N * 6 * max_h * max_w;
    if ((int)batch_tensor_data.size() < needed) {
        batch_tensor_data.resize(needed, 0.0f);
    }
    float* batch_data = batch_tensor_data.data();
    
    // Zero solo la porción que vamos a usar (Fix 2: no zeroing innecesario del buffer completo)
    std::fill(batch_data, batch_data + needed, 0.0f);
    
    // Default background es Wall (Channel 0 = 1.0) para todos los nodos
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

    auto input_tensor = torch::from_blob(batch_data, {MAX_BATCH_SIZE, 6, max_h, max_w}, torch::kFloat32);
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

    // Fix 1: mover tensor a CPU UNA sola vez, luego leer con accessor (sin sync GPU por cada elemento)
    auto cpu_tensor = pushes_tensor.to(torch::kCPU).contiguous();
    auto accessor = cpu_tensor.accessor<float, 1>();
    static std::ofstream dump_file("scratch/classifier_dump.txt", std::ios::app);
    for (size_t i = 0; i < nodes.size(); i++) {
        dump_file << nodes[i]->board_to_pretty_string() << "===";
    }
    dump_file << "\n";
    dump_file.flush();

    std::vector<float> results;
    results.reserve(N);
    for (int i = 0; i < N; ++i) {
        float z_score = accessor[i];  // sin sincronización GPU individual
        float pushes_pred = std::expm1(z_score * pushes_std + pushes_mean);
        
        // Explicit numerical safety guard
        if (!std::isfinite(pushes_pred) || pushes_pred > 1e6f || pushes_pred < 0.0f) {
            pushes_pred = 1e6f;
        }
        
        // Apply Isotonic Calibration
        pushes_pred = apply_calibration(pushes_pred);
        
        if (dist_initialized && !goal_positions.empty()) {
            const game_node* node = nodes[i];
            int num_boxes = node->box_count;
            int num_goals = (int)goal_positions.size();
            int sz = std::max(num_boxes, num_goals);
            std::vector<std::vector<int>> cost(sz, std::vector<int>(sz, 0));
            for (int b = 0; b < num_boxes; ++b) {
                for (int g = 0; g < num_goals; ++g) {
                    cost[b][g] = dist_to_goal[g][node->box_list[b].x][node->box_list[b].y];
                }
            }
            Hungarian h(cost);
            float hungarian_lb = (float)h.solve();
            float clipped_val = hungarian_lb + std::clamp(pushes_pred - hungarian_lb, 0.0f, 1.318f * hungarian_lb);
            std::cout << "[DEBUG] C++ batch - raw: " << z_score << " -> expm1: " << (float)std::expm1(z_score * pushes_std + pushes_mean) << " -> calib: " << pushes_pred << " -> clip(1.318x): " << clipped_val << std::endl;
            pushes_pred = clipped_val;
        }
        
        results.push_back(pushes_pred);
    }

    return results;
}

void NeuralHeuristic::reset_board(int rows, int cols) {
    this->m = rows;
    this->n = cols;
    this->mask_initialized = false;
    this->dist_initialized = false;
    this->goal_positions.clear();
    this->dist_to_goal.clear();
}
