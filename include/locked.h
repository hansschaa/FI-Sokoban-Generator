#pragma once

#include "point.h"
#include <vector>

class locked {
private:
    std::vector<std::vector<bool>> side_point;
    bool locked_double(std::vector<std::vector<char>>& matrix_with_box, point& box, point& wall);
    std::vector<point> get_box_wall(point &box);
    bool is_next_two_wall(std::vector<point> &around);

    private:
    // Soporte recursivo
    bool check_box_frozen_rec(point box, const std::vector<std::vector<char>>& matrix, std::vector<point>& visited);
    bool is_axis_blocked(point box, int dx, int dy, const std::vector<std::vector<char>>& matrix, std::vector<point>& visited);

public:
    locked();
    void init();
    bool is_locked(point& box, std::vector<std::vector<char>>& matrix_with_box);
    
    // Nuevo: Chequeo rápido de Freeze Deadlock
    bool is_freeze_deadlock(const point& box, const std::vector<std::vector<char>>& matrix_with_box);

    const std::vector<std::vector<bool>>& get_side_point() const { return side_point; }
};
