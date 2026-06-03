#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <iostream>

struct PathBranchingStats {
    int branching_real_total_nodes = 0;
    int branching_real_min = 2147483647;
    int branching_real_max = -2147483648;

    int branching_effective_total_nodes = 0;
    int branching_effective_min = 2147483647;
    int branching_effective_max = -2147483648;

    int states = 0;
    int total_children_generated = 0;
    long repeated_nodes = 0;
    int deadlocks = 0;

    double get_branching_real_avg() const {
        return states == 0 ? 0.0 : (double)branching_real_total_nodes / states;
    }

    double get_branching_effective_avg() const {
        return states == 0 ? 0.0 : (double)branching_effective_total_nodes / states;
    }

    double get_redundancy() const {
        return repeated_nodes == 0 ? 0.0 : (double)total_children_generated / repeated_nodes;
    }
};

class PathSimulator {
private:
    struct SimState {
        std::vector<std::string> grid;
        int w, h;

        SimState(const std::string& flat_board, int rows, int cols);
        SimState(const SimState& other) = default;

        std::vector<char> get_legal_movements() const;
        std::unique_ptr<SimState> apply_move(char m) const;
        std::string to_string_key() const;
        bool is_pattern2_deadlock() const;
    };

public:
    static PathBranchingStats compute_stats(const std::string& flat_initial_board, int rows, int cols, const std::string& lurd);
};