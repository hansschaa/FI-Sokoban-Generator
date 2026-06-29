#include "game_node.h"
#include "mazesolver.h"
#include "constant.h"
#include "repeat.h"

using namespace constant;
#include <algorithm>

using namespace constant;
using namespace std;

game_node::game_node(const point* boxes, int8_t count, point& ps) {
    box_count = count;
    for(int i = 0; i < count; i++) {
        box_list[i] = boxes[i];
    }
    std::sort(box_list, box_list + box_count);
    person_point = ps;
    normalize_player();
}

game_node::game_node(){}

void game_node::get_matrix0(vector<vector<char>>& result)const {
    result = blank_matrix;
    for (int i = 0; i < box_count; i++) {
        result[box_list[i].x][box_list[i].y] = BOX;
    }
}

vector<vector<char>> game_node::get_matrix()const {
    auto result = blank_matrix;

    for (int i = 0; i < box_count; i++) {
        result[box_list[i].x][box_list[i].y] = BOX;
    }
    return result;
}

vector<vector<char>> game_node::get_matrix2()const {
    vector<vector<char>> result;
    result = blank_matrix;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (end_vec[i][j] == true) {
                result[i][j] = FINAL;
            }
        }
    }
    for (int i = 0; i < box_count; i++) {
        bool onGoal = end_vec[box_list[i].x][box_list[i].y];
        result[box_list[i].x][box_list[i].y] = onGoal ? REDBOX : BOX;
    }

    bool playerOnGoal = end_vec[person_point.x][person_point.y];
    result[person_point.x][person_point.y] = playerOnGoal ? PERSONF : PERSON;
    return result;
}

bool game_node::operator==(const game_node &a)const {
    if (a.box_count != box_count || !(a.person_point == person_point)) return false;
    for (int i = 0; i < box_count; i++) {
        if (!(a.box_list[i] == box_list[i])) return false;
    }
    return true;
}

bool game_node::game_over() const {
    for (int i = 0; i < box_count; i++) {
        auto p = box_list[i];
        if (end_vec[p.x][p.y] == false) {return false;}
    }
    return true;
}

void game_node::get_moved(const point& box_before, point& box_new, game_node* result) const {
    *result = *this;
    for (int i = 0; i < result->box_count; i++) {
        if (result->box_list[i] == box_before) {
            result->box_list[i] = box_new;
            break;
        }
    }
    std::sort(result->box_list, result->box_list + result->box_count);
    result->person_point = box_before;
    result->normalize_player();

    // Actualización incremental del hash Zobrist
    if (this->hash_calculated) {
        result->cached_hash = this->cached_hash ^ repeat::zobrist[box_before.x][box_before.y] ^ repeat::zobrist[box_new.x][box_new.y];
        result->hash_calculated = true;
    } else {
        result->hash_calculated = false;
    }
}

size_t game_node::get_hash() const {
    if (!hash_calculated) {
        size_t result = 0;
        for (int i = 0; i < box_count; i++) {
            result = result ^ repeat::zobrist[box_list[i].x][box_list[i].y];
        }
        cached_hash = result;
        hash_calculated = true;
    }
    return cached_hash;
}

void game_node::normalize_player() {
    bool blocked[128][128] = {false};
    int rows = constant::m;
    int cols = constant::n;
    if (rows > 128) rows = 128;
    if (cols > 128) cols = 128;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            blocked[r][c] = (constant::blank_matrix[r][c] == constant::WALL);
        }
    }
    for (int i = 0; i < box_count; i++) {
        auto box = box_list[i];
        if (box.x >= 0 && box.x < rows && box.y >= 0 && box.y < cols) {
            blocked[box.x][box.y] = true;
        }
    }

    std::vector<point> q;
    q.reserve(rows * cols);
    int head = 0;
    bool visited[128][128] = {false};
    if (person_point.x >= 0 && person_point.x < rows && person_point.y >= 0 && person_point.y < cols) {
        q.push_back(person_point);
        visited[person_point.x][person_point.y] = true;
    }
    point min_p = person_point;
    while (head < (int)q.size()) {
        point curr = q[head++];
        if (curr < min_p) {
            min_p = curr;
        }
        for (auto& direction : constant::four_direction) {
            point next = curr + direction;
            if (next.x >= 0 && next.x < rows && next.y >= 0 && next.y < cols) {
                if (!blocked[next.x][next.y] && !visited[next.x][next.y]) {
                    visited[next.x][next.y] = true;
                    q.push_back(next);
                }
            }
        }
    }
    person_point = min_p;
}