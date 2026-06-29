#pragma once

#include "point.h"
#include <cstddef>
#include <vector>
#include <set>

class game_node {
public:
    point box_list[16];
    int8_t box_count = 0;
    point person_point;
    mutable size_t cached_hash = 0;
    mutable bool hash_calculated = false;

    game_node(const point* boxes, int8_t count, point& ps);
    game_node();
    void get_matrix0(std::vector<std::vector<char>>& result)const;
    std::vector<std::vector<char>> get_matrix()const;
    std::vector<std::vector<char>> get_matrix2()const;
    bool operator==(const game_node &a)const;
    bool game_over()const;
    void get_moved(const point& box_before, point& box_new,game_node* result)const;
    size_t get_hash() const;
    void normalize_player();
};