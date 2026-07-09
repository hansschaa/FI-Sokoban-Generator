#pragma once

#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>

struct StructuralFeatures {
    double wall_density = 0.0;
    double open_space_ratio = 0.0;
    int connectivity = 0;
    double aspect_ratio = 1.0;
    double dead_end_ratio = 0.0;
    double avg_symmetry = 0.0;
    int num_interior_regions = 0;
};

class StructuralMetricsCalculator {
private:
    struct Point {
        int r, c;
        bool operator<(const Point& other) const {
            if (r != other.r) return r < other.r;
            return c < other.c;
        }
        bool operator==(const Point& other) const {
            return r == other.r && c == other.c;
        }
    };

    static void find_bounding_box(const std::vector<std::vector<char>>& grid, int& min_r, int& max_r, int& min_c, int& max_c) {
        int rows = grid.size();
        int cols = grid[0].size();
        min_r = rows; max_r = -1;
        min_c = cols; max_c = -1;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == '#') {
                    if (r < min_r) min_r = r;
                    if (r > max_r) max_r = r;
                    if (c < min_c) min_c = c;
                    if (c > max_c) max_c = c;
                }
            }
        }
        if (max_r < min_r) { // No walls
            min_r = 0; max_r = rows - 1;
            min_c = 0; max_c = cols - 1;
        }
    }

    static std::set<Point> get_interior_cells(const std::vector<std::vector<char>>& grid, std::vector<std::set<Point>>& out_regions) {
        int rows = grid.size();
        int cols = grid[0].size();
        std::set<Point> exterior;
        std::queue<Point> q;

        // Flood fill from edges
        for (int r = 0; r < rows; r++) {
            if (grid[r][0] == ' ') { exterior.insert({r, 0}); q.push({r, 0}); }
            if (grid[r][cols - 1] == ' ') { exterior.insert({r, cols - 1}); q.push({r, cols - 1}); }
        }
        for (int c = 0; c < cols; c++) {
            if (grid[0][c] == ' ') { exterior.insert({0, c}); q.push({0, c}); }
            if (grid[rows - 1][c] == ' ') { exterior.insert({rows - 1, c}); q.push({rows - 1, c}); }
        }

        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        while (!q.empty()) {
            Point p = q.front();
            q.pop();
            for (int i = 0; i < 4; i++) {
                int nr = p.r + dr[i];
                int nc = p.c + dc[i];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    if (grid[nr][nc] == ' ' && exterior.find({nr, nc}) == exterior.end()) {
                        exterior.insert({nr, nc});
                        q.push({nr, nc});
                    }
                }
            }
        }

        std::set<Point> candidates;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == ' ' && exterior.find({r, c}) == exterior.end()) {
                    candidates.insert({r, c});
                }
            }
        }

        std::set<Point> visited;
        std::set<Point> largest_region;

        for (const Point& start : candidates) {
            if (visited.find(start) == visited.end()) {
                std::set<Point> region;
                std::queue<Point> q2;
                visited.insert(start);
                q2.push(start);
                while (!q2.empty()) {
                    Point p = q2.front();
                    q2.pop();
                    region.insert(p);
                    for (int i = 0; i < 4; i++) {
                        int nr = p.r + dr[i];
                        int nc = p.c + dc[i];
                        Point np = {nr, nc};
                        if (candidates.find(np) != candidates.end() && visited.find(np) == visited.end()) {
                            visited.insert(np);
                            q2.push(np);
                        }
                    }
                }
                out_regions.push_back(region);
                if (region.size() > largest_region.size()) {
                    largest_region = region;
                }
            }
        }
        return largest_region;
    }

    static std::set<Point> get_inner_walls(const std::vector<std::vector<char>>& grid, const std::set<Point>& interior) {
        std::set<Point> inner_walls;
        int rows = grid.size();
        int cols = grid[0].size();
        int dr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int dc[] = {0, 0, -1, 1, -1, 1, -1, 1};
        
        for (const Point& p : interior) {
            for (int i = 0; i < 8; i++) {
                int nr = p.r + dr[i];
                int nc = p.c + dc[i];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                    if (grid[nr][nc] == '#') {
                        inner_walls.insert({nr, nc});
                    }
                }
            }
        }
        return inner_walls;
    }

    static double horizontal_symmetry(const std::vector<std::vector<char>>& grid, int min_r, int max_r, int min_c, int max_c) {
        int matches = 0, total = 0;
        for (int r = min_r; r <= max_r; r++) {
            for (int c = min_c; c <= max_c; c++) {
                int mirror_c = max_c - (c - min_c);
                if (mirror_c <= max_c) {
                    total++;
                    if (grid[r][c] == grid[r][mirror_c]) matches++;
                }
            }
        }
        return total > 0 ? (double)matches / total : 0.0;
    }

    static double vertical_symmetry(const std::vector<std::vector<char>>& grid, int min_r, int max_r, int min_c, int max_c) {
        int matches = 0, total = 0;
        for (int r = min_r; r <= max_r; r++) {
            int mirror_r = max_r - (r - min_r);
            for (int c = min_c; c <= max_c; c++) {
                if (mirror_r <= max_r) {
                    total++;
                    if (grid[r][c] == grid[mirror_r][c]) matches++;
                }
            }
        }
        return total > 0 ? (double)matches / total : 0.0;
    }

public:
    static StructuralFeatures calculate(std::vector<std::vector<char>> grid) {
        // Convert dynamic objects to floor space
        int rows = grid.size();
        int cols = grid[0].size();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                char ch = grid[r][c];
                if (ch == '@' || ch == '$' || ch == '.' || ch == '*' || ch == '+') {
                    grid[r][c] = ' ';
                }
            }
        }

        StructuralFeatures f;
        
        int min_r, max_r, min_c, max_c;
        find_bounding_box(grid, min_r, max_r, min_c, max_c);
        
        int bb_total = (max_r - min_r + 1) * (max_c - min_c + 1);
        int bb_walls = 0;
        for (int r = min_r; r <= max_r; r++) {
            for (int c = min_c; c <= max_c; c++) {
                if (grid[r][c] == '#') bb_walls++;
            }
        }
        f.wall_density = bb_total > 0 ? (double)bb_walls / bb_total : 0.0;
        
        std::vector<std::set<Point>> regions;
        std::set<Point> interior = get_interior_cells(grid, regions);
        
        std::set<Point> inner_walls = get_inner_walls(grid, interior);
        int denom_inner = interior.size() + inner_walls.size();
        f.open_space_ratio = denom_inner > 0 ? (double)interior.size() / denom_inner : 0.0;
        
        f.connectivity = interior.size();
        f.aspect_ratio = (max_r - min_r + 1) > 0 ? (double)(max_c - min_c + 1) / (max_r - min_r + 1) : 1.0;
        
        f.num_interior_regions = regions.size();
        
        f.avg_symmetry = (horizontal_symmetry(grid, min_r, max_r, min_c, max_c) + vertical_symmetry(grid, min_r, max_r, min_c, max_c)) / 2.0;
        
        int dead_ends = 0;
        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};
        for (const Point& p : interior) {
            int free_neighbors = 0;
            for (int i = 0; i < 4; i++) {
                Point np = {p.r + dr[i], p.c + dc[i]};
                if (interior.find(np) != interior.end()) {
                    free_neighbors++;
                }
            }
            if (free_neighbors == 1) {
                dead_ends++;
            }
        }
        f.dead_end_ratio = interior.size() > 0 ? (double)dead_ends / interior.size() : 0.0;
        
        return f;
    }
};
