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
    std::vector<const Node*> discarded_nodes;

    bool did_timeout() const {
        return timeout_reached;
    }
    Result solve(
        const Node* start, const Node* goal,
        std::function<void(const Node*, std::function<void(const Node*)>)> get_neighbors,
        std::function<bool(const Node*)> is_visited,
        std::function<void(const Node*)> mark_visited,
        std::function<bool(const Node*, const Node*)> is_equal,
        std::function<int(const Node*, const Node*)> heuristic = nullptr,
        std::function<std::vector<int>(const std::vector<const Node*>&, const Node*)> heuristic_batch = nullptr,
        int batch_k = 1,           // 1 = per-node batch  |  >1 = cross-node batch (acumula batch_k nodos padres)
        double max_seconds = 60.0,
        size_t max_nodes = 2000000
    ) {
        h_cache.clear();
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
        size_t loop_count = 0;
        
        // Ajuste dinámico del chequeo de timeout: si procesamos muchos nodos por iteración,
        // chequear más seguido para no exceder max_seconds en exceso.
        int check_interval = std::max(1, 1024 / batch_k);

        while (!container.empty()) {
            loop_count++;
            if (loop_count % check_interval == 0) {
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
                    // Lo guardamos en discarded_nodes para que el caller pueda
                    // hacer deallocate() en su memory pool.
                    discarded_nodes.push_back(current);
                    continue;
                }
                mark_visited(current);
            }

            bool found = false;

            std::vector<const Node*> children;
            get_neighbors(current, [&](const Node* neighbor) {
                if (is_visited(neighbor)) {
                    discarded_nodes.push_back(neighbor);
                    return;
                }
                children.push_back(neighbor);
            });

            if (!children.empty()) {
                if constexpr (alg == Method::a_star) {
                    if (heuristic_batch) {
                        if (batch_k > 1) {
                            // ── CROSS-NODE BATCHING ──────────────────────────────────
                            // Acumula hijos de hasta batch_k nodos padres antes de
                            // una sola llamada GPU. Batch grande → GPU más eficiente.
                            struct PendingChild {
                                const Node* child;
                                int          g_parent;
                                const Node*  parent_ptr;
                            };
                            std::vector<PendingChild> pending;
                            pending.reserve(batch_k * 8);

                            for (auto child : children) {
                                pending.push_back({child, g_current, current});
                            }

                            // Expandir hasta batch_k-1 nodos adicionales
                            for (int k = 1; k < batch_k && !container.empty(); ) {
                                auto [f2, g2, node2] = container.top();
                                container.pop();

                                if (is_visited(node2)) {
                                    discarded_nodes.push_back(node2);
                                    continue;
                                }
                                mark_visited(node2);
                                k++;

                                const int g2_local = g2;
                                const Node* node2_local = node2;
                                get_neighbors(node2_local, [&](const Node* neighbor) {
                                    if (is_visited(neighbor)) {
                                        discarded_nodes.push_back(neighbor);
                                        return;
                                    }
                                    pending.push_back({neighbor, g2_local, node2_local});
                                });
                            }

                            // Deduplicate `pending` using `h_cache` and `local_pending`
                            std::vector<int> h_scores(pending.size(), -1);
                            std::vector<const Node*> to_evaluate;
                            std::unordered_map<size_t, std::vector<size_t>> local_pending_indices;

                            for (size_t i = 0; i < pending.size(); i++) {
                                size_t hash = pending[i].child->get_hash();
                                auto it = h_cache.find(hash);
                                if (it != h_cache.end()) {
                                    h_scores[i] = it->second;
                                } else {
                                    auto local_it = local_pending_indices.find(hash);
                                    if (local_it != local_pending_indices.end()) {
                                        local_pending_indices[hash].push_back(i);
                                    } else {
                                        local_pending_indices[hash] = {i};
                                        to_evaluate.push_back(pending[i].child);
                                    }
                                }
                            }

                            if (!to_evaluate.empty()) {
                                auto evaluated = heuristic_batch(to_evaluate, goal);
                                for (size_t i = 0; i < to_evaluate.size(); i++) {
                                    size_t hash = to_evaluate[i]->get_hash();
                                    int score = evaluated[i];
                                    h_cache[hash] = score;
                                    for (size_t idx : local_pending_indices[hash]) {
                                        h_scores[idx] = score;
                                    }
                                }
                            }

                            for (size_t i = 0; i < pending.size(); i++) {
                                const auto& [child, g_par, par] = pending[i];
                                generated_count++;

                                if constexpr (!std::is_same_v<Result, bool>) {
                                    if (parent.find(child) == parent.end()) {
                                        parent[child] = par;
                                    }
                                }

                                int g_child = g_par + 1;
                                int f_score = g_child + h_scores[i];
                                container.push({f_score, g_child, child});

                                if (is_equal(child, goal)) {
                                    found = true;
                                    // NOTA (Auditoría): Esto reasigna 'goal' a mitad del batch, mientras aún
                                    // quedan hijos por evaluar. Hoy es inofensivo porque nuestro is_equal() 
                                    // ignora su 2do parámetro (solo chequea child->game_over()), pero si 
                                    // en el futuro is_equal llegara a comparar explícitamente contra 'goal', 
                                    // esto se activaría como un bug real de estado inconsistente.
                                    goal  = child;
                                }
                            }

                        } else {
                            // ── PER-NODE BATCHING (batch_k=1) ────────────────────────
                            std::vector<int> h_scores(children.size(), -1);
                            std::vector<const Node*> to_evaluate;
                            std::unordered_map<size_t, std::vector<size_t>> local_pending_indices;

                            for (size_t i = 0; i < children.size(); i++) {
                                size_t hash = children[i]->get_hash();
                                auto it = h_cache.find(hash);
                                if (it != h_cache.end()) {
                                    h_scores[i] = it->second;
                                } else {
                                    auto local_it = local_pending_indices.find(hash);
                                    if (local_it != local_pending_indices.end()) {
                                        local_pending_indices[hash].push_back(i);
                                    } else {
                                        local_pending_indices[hash] = {i};
                                        to_evaluate.push_back(children[i]);
                                    }
                                }
                            }

                            if (!to_evaluate.empty()) {
                                auto evaluated = heuristic_batch(to_evaluate, goal);
                                for (size_t i = 0; i < to_evaluate.size(); i++) {
                                    size_t hash = to_evaluate[i]->get_hash();
                                    int score = evaluated[i];
                                    h_cache[hash] = score;
                                    for (size_t idx : local_pending_indices[hash]) {
                                        h_scores[idx] = score;
                                    }
                                }
                            }

                            for (size_t i = 0; i < children.size(); i++) {
                                const Node* neighbor = children[i];
                                generated_count++;

                                if constexpr (!std::is_same_v<Result, bool>) {
                                    if (parent.find(neighbor) == parent.end()) {
                                        parent[neighbor] = current;
                                    }
                                }

                                int g_neighbor = g_current + 1;
                                int f_score    = g_neighbor + h_scores[i];
                                container.push({f_score, g_neighbor, neighbor});

                                if (is_equal(neighbor, goal)) {
                                    found = true;
                                    goal  = neighbor;
                                }
                            }
                        }

                    } else {
                        // ── HEURÍSTICA ESCALAR (sin batch) ──────────────────────────
                        std::vector<int> h_scores;
                        h_scores.reserve(children.size());
                        for (auto n : children) {
                            size_t hash = n->get_hash();
                            auto it = h_cache.find(hash);
                            if (it != h_cache.end()) {
                                h_scores.push_back(it->second);
                            } else {
                                int score = heuristic(n, goal);
                                h_cache[hash] = score;
                                h_scores.push_back(score);
                            }
                        }

                        for (size_t i = 0; i < children.size(); i++) {
                            const Node* neighbor = children[i];
                            generated_count++;

                            if constexpr (!std::is_same_v<Result, bool>) {
                                if (parent.find(neighbor) == parent.end()) {
                                    parent[neighbor] = current;
                                }
                            }

                            int g_neighbor = g_current + 1;
                            int f_score    = g_neighbor + h_scores[i];
                            container.push({f_score, g_neighbor, neighbor});

                            if (is_equal(neighbor, goal)) {
                                found = true;
                                goal  = neighbor;
                            }
                        }
                    }
                }
                else {
                    for (const Node* neighbor : children) {
                        generated_count++;
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
                }
            }

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

    std::unordered_map<size_t, int> h_cache;

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