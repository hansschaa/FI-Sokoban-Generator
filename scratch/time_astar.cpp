#include "../include/game_solver.h"
#include <iostream>

int main() {
    std::string board = "##############\n#        $   #\n#  #  $ #    #\n#  $  $ $ $  #\n#  $     @   #\n#     $      #\n##############";
    game_solver solver(board, 7, 14, 64);
    solver.enable_advanced_deadlocks = true;
    std::vector<game_node> solution;
    auto stats = solver.test_template(Method::a_star, Heuristic::hungarian, solution, false, nullptr, 120.0, 500000);
    std::cout << "Time: " << stats.runtime_ms << " ms\n";
    std::cout << "Nodes: " << stats.expanded_nodes << "\n";
    std::cout << "Status: " << (int)stats.status << "\n";
    return 0;
}
