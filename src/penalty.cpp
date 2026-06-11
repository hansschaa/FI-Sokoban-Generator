#include "../include/penalty.h"
#include "../include/constant.h"
#include <algorithm>

Penalty::Penalty(int mm, int nn) : m(mm), n(nn), box_squares_count(0) {}

void Penalty::init(
    const std::vector<point>& goal_positions,
    const std::vector<std::vector<std::vector<int>>>& dist_to_goal,
    const std::vector<std::vector<bool>>& side_point
) {
    penalty_situations.clear();
    if (m * n > 4900) return;

    // 1. Identificar posiciones relevantes para cajas
    board_squares_to_box_squares.assign(m * n, -1);
    std::vector<short> temp_box_squares_to_board_squares(m * n, -1);
    box_squares_count = 0;

    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            int pos = x * n + y;
            if (constant::blank_matrix[x][y] != constant::WALL && !side_point[x][y]) {
                board_squares_to_box_squares[pos] = box_squares_count;
                temp_box_squares_to_board_squares[box_squares_count++] = pos;
            }
        }
    }

    box_squares_to_board_squares.assign(
        temp_box_squares_to_board_squares.begin(),
        temp_box_squares_to_board_squares.begin() + box_squares_count
    );

    // 2. Identificar potenciales casillas de penalización
    const int UP_DIR = 3;
    const int DOWN_DIR = 1;
    const int LEFT_DIR = 2;
    const int RIGHT_DIR = 0;

    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            int pos = x * n + y;
            if (constant::blank_matrix[x][y] == constant::WALL || side_point[x][y]) {
                continue;
            }

            // --- CHEQUEO HORIZONTAL (Caja a la derecha: pos + 1) ---
            int neighbor_h = pos + 1;
            int nh_x = neighbor_h / n;
            int nh_y = neighbor_h % n;

            if (nh_x == x && nh_y < n && constant::blank_matrix[nh_x][nh_y] != constant::WALL && !side_point[nh_x][nh_y]) {
                bool both_goals = constant::end_vec[x][y] && constant::end_vec[nh_x][nh_y];
                if (!both_goals) {
                    bool can_pos_up = is_push_without_lowerbound_increase_possible(pos, UP_DIR, goal_positions, dist_to_goal, side_point);
                    bool can_nh_up  = is_push_without_lowerbound_increase_possible(neighbor_h, UP_DIR, goal_positions, dist_to_goal, side_point);

                    if (!can_pos_up && !can_nh_up) {
                        PenaltySituation sit;
                        sit.pos1 = pos;
                        sit.pos2 = neighbor_h;
                        sit.is_player_dependent = false;
                        penalty_situations.push_back(sit);
                    }
                }
            }

            // --- CHEQUEO VERTICAL (Caja abajo: pos + n) ---
            int neighbor_v = pos + n;
            int nv_x = neighbor_v / n;
            int nv_y = neighbor_v % n;

            if (nv_x < m && constant::blank_matrix[nv_x][nv_y] != constant::WALL && !side_point[nv_x][nv_y]) {
                bool both_goals = constant::end_vec[x][y] && constant::end_vec[nv_x][nv_y];
                if (!both_goals) {
                    bool can_pos_right = is_push_without_lowerbound_increase_possible(pos, RIGHT_DIR, goal_positions, dist_to_goal, side_point);
                    bool can_nv_right  = is_push_without_lowerbound_increase_possible(neighbor_v, RIGHT_DIR, goal_positions, dist_to_goal, side_point);

                    if (!can_pos_right && !can_nv_right) {
                        PenaltySituation sit;
                        sit.pos1 = pos;
                        sit.pos2 = neighbor_v;
                        sit.is_player_dependent = false;
                        penalty_situations.push_back(sit);
                    }
                }
            }
        }
    }
}

bool Penalty::is_push_without_lowerbound_increase_possible(
    int boxPosition, 
    int direction,
    const std::vector<point>& goal_positions,
    const std::vector<std::vector<std::vector<int>>>& dist_to_goal,
    const std::vector<std::vector<bool>>& side_point
) const {
    int pos_dir = get_position(boxPosition, direction);
    int pos_opp = get_position_opposite(boxPosition, direction);

    if (is_wall(pos_dir) || is_wall(pos_opp)) {
        return false;
    }
    
    int pd_x = pos_dir / n, pd_y = pos_dir % n;
    int po_x = pos_opp / n, po_y = pos_opp % n;
    if (side_point[pd_x][pd_y] && side_point[po_x][po_y]) {
        return false;
    }

    int bx = boxPosition / n;
    int by = boxPosition % n;

    int num_goals = (int)goal_positions.size();
    for (int g = 0; g < num_goals; g++) {
        int dist_curr = dist_to_goal[g][bx][by];
        if (dist_curr >= 1000) continue;

        int dist_dir = dist_to_goal[g][pd_x][pd_y];
        int dist_opp = dist_to_goal[g][po_x][po_y];

        if (dist_dir < dist_curr || dist_opp < dist_curr) {
            return true;
        }
    }

    return false;
}

int Penalty::get_position(int pos, int dir) const {
    int x = pos / n;
    int y = pos % n;
    point p(x, y);
    point new_p = p + constant::four_direction[dir];
    if (new_p.x < 0 || new_p.x >= m || new_p.y < 0 || new_p.y >= n) {
        return -1;
    }
    return new_p.x * n + new_p.y;
}

int Penalty::get_position_opposite(int pos, int dir) const {
    int x = pos / n;
    int y = pos % n;
    point p(x, y);
    point new_p = p - constant::four_direction[dir];
    if (new_p.x < 0 || new_p.x >= m || new_p.y < 0 || new_p.y >= n) {
        return -1;
    }
    return new_p.x * n + new_p.y;
}

bool Penalty::is_wall(int pos) const {
    if (pos < 0 || pos >= m * n) return true;
    int x = pos / n;
    int y = pos % n;
    return constant::blank_matrix[x][y] == constant::WALL;
}

int Penalty::calculate_penalty(const std::set<point>& box_list) const {
    if (m * n > 4900 || penalty_situations.empty()) {
        return 0;
    }

    std::vector<bool> current_situation(box_squares_count, false);
    for (const auto& box : box_list) {
        int pos = box.x * n + box.y;
        if (pos >= 0 && pos < m * n) {
            int box_sq = board_squares_to_box_squares[pos];
            if (box_sq >= 0 && box_sq < box_squares_count) {
                current_situation[box_sq] = true;
            }
        }
    }

    int penalty = 0;
    for (const auto& sit : penalty_situations) {
        int idx1 = board_squares_to_box_squares[sit.pos1];
        int idx2 = board_squares_to_box_squares[sit.pos2];

        if (idx1 >= 0 && idx1 < box_squares_count && current_situation[idx1] &&
            idx2 >= 0 && idx2 < box_squares_count && current_situation[idx2]) {
            
            penalty += 2;
            current_situation[idx1] = false;
            current_situation[idx2] = false;
        }
    }

    return penalty;
}
