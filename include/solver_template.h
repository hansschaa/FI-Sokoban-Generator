#pragma once
#include "method.h"   // FIX: Method debe declararse ANTES del template que lo usa
#include <queue>
#include <vector>
#include <functional>
#include <algorithm>
#include <stack>
#include <chrono>
#include <unordered_map>
#include <tuple>

template<typename ResultType, typename NodeType, Method alg>
class Solver_template {

bool timeout_reached = false;

public:
    using Node = NodeType;
    using Result = ResultType;

    // Nodos que quedaron en la open list sin expandir al momento de terminar.
    // El caller debe destruirlos y desalocarlos antes del siguiente run.
    std::vector<const Node*> orphan_nodes;

    bool did_timeout() const {
        return timeout_reached;
    }
    Result solve (
    const Node* start,
    const Node* goal,
    std::function<void(const Node*, std::function<void(const Node*)>)> get_neighbors,
    std::function<bool(const Node*)> is_visited,
    std::function<void(const Node*)> mark_visited,
    std::function<bool(const Node*, const Node*)> is_equal,
    std::function<int(const Node*, const Node*)> heuristic = nullptr,
    double max_seconds = 60.0,
    size_t max_nodes = 50000
    ) {
        if (is_equal(start, goal)) {
            if constexpr (std::is_same_v<Result, bool>) {
                return true;
            }
            else if constexpr (std::is_same_v<Result, std::vector<Node>>) {
                return {*start};
            }
        }

        using Container_Type = std::conditional_t<
                alg == Method::bfs,
                std::queue<const Node*>,
                std::conditional_t<alg == Method::dfs,
                std::stack<const Node*>,
                std::priority_queue<
                    std::tuple<int, int, const Node*>,
                    std::vector<std::tuple<int, int, const Node*>>,
                    decltype(&Solver_template::compare_tuples)>>
                >;

        Container_Type container;

        if constexpr (alg == Method::a_star) {
            container = std::priority_queue<
                std::tuple<int, int, const Node*>,
                std::vector<std::tuple<int, int, const Node*>>,
                decltype(&Solver_template::compare_tuples)>(&Solver_template::compare_tuples);
        }

        std::unordered_map<const Node*, const Node*> parent;

        if constexpr (alg == Method::a_star) {
            container.push({heuristic(start, goal), 0, start});
        }
        else {
            container.push(start);
            mark_visited(start);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t generated_count = 0;
        while (!container.empty()) {

            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();

            if (elapsed > max_seconds || generated_count > max_nodes) {
                timeout_reached = true;
                // Drenar la open list en caso de TIMEOUT
                if constexpr (alg == Method::a_star) {
                    while (!container.empty()) {
                        auto [f, g, node] = container.top();
                        container.pop();
                        orphan_nodes.push_back(node);
                    }
                }
                return get_default_result();
            }

            const Node* current = nullptr;
            int g_current = 0;

            if constexpr (alg == Method::bfs) {
                current = container.front();
            }
            else if constexpr (alg == Method::dfs) {
                current = container.top();
            }
            else {
                auto [f, g, node] = container.top();
                current   = node;
                g_current = g;
            }
            container.pop();

            if constexpr (alg == Method::a_star) {
                if (is_visited(current)) {
                    // --- EL PARCHE DEFINITIVO ---
                    // Este nodo es un duplicado ya visitado. Al descartarlo con 'continue',
                    // sale de la Open List pero no entra a la Closed List. 
                    // DEBEMOS llamar a su destructor explícitamente para liberar el heap de su std::set.
                    const_cast<Node*>(current)->~Node();
                    continue;
                }
                mark_visited(current);
            }

            bool found = false;

            get_neighbors(current, [&](const Node* neighbor) {
                generated_count++;
                if constexpr (alg == Method::a_star) {
                    if constexpr (!std::is_same_v<Result, bool>) {
                        if (parent.find(neighbor) == parent.end()) {
                            parent[neighbor] = current;
                        }
                    }

                    int g_neighbor = g_current + 1;
                    int f_score    = g_neighbor + heuristic(neighbor, goal);
                    container.push({f_score, g_neighbor, neighbor});

                    if (is_equal(neighbor, goal)) {
                        found = true;
                        goal  = neighbor;
                    }
                }
                else {
                    if (!is_visited(neighbor)) {
                        mark_visited(neighbor);
                        if constexpr (!std::is_same_v<Result, bool>) {
                            parent[neighbor] = current;
                        }
                        container.push(neighbor);

                        if (is_equal(neighbor, goal)) {
                            found = true;
                            goal  = neighbor;
                        }
                    }
                }
            });

            if (found) {
                // --- ARREGLO DE LEAK: Drenar la open list antes de salir con éxito (SOLVED) ---
                if constexpr (alg == Method::a_star) {
                    while (!container.empty()) {
                        auto [f, g, node] = container.top();
                        container.pop();
                        orphan_nodes.push_back(node);
                    }
                }
                
                if constexpr (std::is_same_v<Result, bool>) {
                    return true;
                }
                else if constexpr (std::is_same_v<Result, std::vector<Node>>) {
                    return get_path(start, goal, parent);
                }
            }
        }

        // --- ARREGLO DE LEAK: Si la lista se vacía sin éxito (UNSOLVABLE), asegurar el vaciado ---
        if constexpr (alg == Method::a_star) {
            while (!container.empty()) {
                auto [f, g, node] = container.top();
                container.pop();
                orphan_nodes.push_back(node);
            }
        }

        return get_default_result();
    }

private:
    std::vector<Node> get_path(const Node* start, const Node* goal,
                               const std::unordered_map<const Node*, const Node*>& parent) {
        std::vector<Node> path;
        const Node* current = goal;
        while (current != start) {
            path.push_back(*current);
            current = parent.at(current);
        }
        path.push_back(*start);
        return path;
    }

    Result get_default_result() {
        if constexpr (std::is_same_v<Result, bool>) {
            return false;
        } else if constexpr (std::is_same_v<Result, std::vector<Node>>) {
            return {};
        }
    }

    static bool compare_tuples(
        const std::tuple<int, int, const Node*>& a,
        const std::tuple<int, int, const Node*>& b)
    {
        return std::get<0>(a) > std::get<0>(b);
    }

    static bool compare_pairs(
        const std::pair<int, const Node*>& a,
        const std::pair<int, const Node*>& b)
    {
        return a.first > b.first;
    }
};