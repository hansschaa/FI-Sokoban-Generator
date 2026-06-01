#pragma once
#include <vector>
#include <algorithm>
#include <limits>

//
// HUNGARIAN ALGORITHM — asignación óptima en O(n³)
// Minimiza la suma total de costos en una matriz n×n cuadrada.
// Si hay más objetivos que cajas (o viceversa), se rellena con ceros.
//
// Uso:
//   Hungarian h(cost_matrix);   // cost_matrix[i][j] = costo caja i → objetivo j
//   int total = h.solve();       // retorna el costo mínimo total
//
class Hungarian {
public:
    explicit Hungarian(const std::vector<std::vector<int>>& cost) {
        int rows = cost.size();
        int cols = rows > 0 ? cost[0].size() : 0;
        n = std::max(rows, cols);

        // Matriz cuadrada rellena con 0
        a.assign(n, std::vector<int>(n, 0));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                a[i][j] = cost[i][j];
    }

    int solve() {
        // u[i], v[j]: potenciales de filas y columnas
        std::vector<int> u(n + 1, 0), v(n + 1, 0);
        // p[j]: qué fila está asignada a la columna j
        std::vector<int> p(n + 1, 0);
        // way[j]: columna previa en el camino aumentante
        std::vector<int> way(n + 1, 0);

        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            std::vector<int> minv(n + 1, std::numeric_limits<int>::max());
            std::vector<bool> used(n + 1, false);

            do {
                used[j0] = true;
                int i0 = p[j0], delta = std::numeric_limits<int>::max(), j1 = -1;

                for (int j = 1; j <= n; j++) {
                    if (!used[j]) {
                        int cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j]  = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1    = j;
                        }
                    }
                }

                for (int j = 0; j <= n; j++) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j]    -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }

                j0 = j1;
            } while (p[j0] != 0);

            do {
                int j1  = way[j0];
                p[j0]   = p[j1];
                j0      = j1;
            } while (j0);
        }

        // Suma de costos de la asignación óptima
        int total = 0;
        for (int j = 1; j <= n; j++)
            if (p[j] != 0)
                total += a[p[j] - 1][j - 1];
        return total;
    }

private:
    int n;
    std::vector<std::vector<int>> a;
};