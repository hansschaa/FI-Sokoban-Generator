#pragma once
// Enum compartido por solver_template.h, mazesolver.h y game_solver.h.
// Vivía implícitamente dentro de solver_template.h pero nunca se declaraba
// antes de usarse en la línea del template, causando errores de compilación
// en cualquier TU que incluyera solver_template.h o mazesolver.h.
enum class Method {
    bfs,
    dfs,
    a_star
};

inline Method int_to_method(int i) {
    switch(i) {
        case 0: return Method::a_star;
        case 1: return Method::dfs;
        case 2: return Method::bfs;
        default: return Method::a_star;
    }
}