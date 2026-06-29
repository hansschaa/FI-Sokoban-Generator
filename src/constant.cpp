#include "constant.h"
using namespace std;

thread_local int8_t constant::m;
thread_local int8_t constant::n;
const point constant::four_direction[4] = { {0, 1}, {1, 0},{0, -1},{-1, 0} };
thread_local my_memory_pool constant::maze_mp;
thread_local my_memory_pool constant::solver_mp;
thread_local vector<vector<char>> constant::blank_matrix;
thread_local vector<vector<bool>> constant::end_vec;
thread_local vector<vector<bool>> constant::matrix0;

bool constant::is_inside(const point &p){
    return p.x >= 0 && p.x <= m - 1 && p.y >= 0 && p.y <= n - 1;
}
