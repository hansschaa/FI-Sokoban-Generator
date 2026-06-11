#pragma once

#include "point.h"
#include <vector>
#include <set>

struct PenaltySituation {
    int pos1;
    int pos2;
    bool is_player_dependent;
};

class Penalty {
private:
    int m;
    int n;
    std::vector<PenaltySituation> penalty_situations;
    std::vector<short> board_squares_to_box_squares;
    std::vector<short> box_squares_to_board_squares;
    short box_squares_count;

    bool is_push_without_lowerbound_increase_possible(
        int boxPosition, 
        int direction,
        const std::vector<point>& goal_positions,
        const std::vector<std::vector<std::vector<int>>>& dist_to_goal,
        const std::vector<std::vector<bool>>& side_point
    ) const;

    int get_position(int pos, int dir) const;
    int get_position_opposite(int pos, int dir) const;
    bool is_wall(int pos) const;

public:
    Penalty(int m = 0, int n = 0);
    
    void init(
        const std::vector<point>& goal_positions,
        const std::vector<std::vector<std::vector<int>>>& dist_to_goal,
        const std::vector<std::vector<bool>>& side_point
    );

    int calculate_penalty(const std::set<point>& box_list) const;
};
