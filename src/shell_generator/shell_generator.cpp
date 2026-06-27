#include "../../include/shell_generator/shell_generator.h"
#include <queue>
#include <iostream>

// Constructor
SokobanGenerator::SokobanGenerator(int fX, int fY) : factorX(fX), factorY(fY) {
    std::random_device rd;
    rng = std::mt19937(rd());
    
    // Tableros con tamaño dinámico + 2 de margen exterior
    // Tableros con tamaño dinámico + 2 de margen exterior (el borde del template)
    width = (factorX * 3) + 2; 
    height = (factorY * 3) + 2;
    board.assign(height, std::vector<Tile>(width, T_IGNORE));
    
    loadBaseTemplates();
}

void SokobanGenerator::generate() {
    bool valid = false;
    int tries = 0;
    while (!valid) {
        tries++;
        if (tries % 1000 == 0) std::cout << "Intento " << tries << "...\n";
        
        board.assign(height, std::vector<Tile>(width, T_IGNORE));
        
        if (!step2_placeTemplates()) {
            if (tries % 1000 == 0) std::cout << "  Fallo step2\n";
            continue; 
        }
        if (!step3_postProcessing()) {
            if (tries % 1000 == 0) {
                if (hasLargeSpaces()) std::cout << "  Fallo step3 (hasLargeSpaces)\n";
                else if (!isConnected()) {
                    std::cout << "  Fallo step3 (!isConnected)\n";
                    std::cout << getBoardString() << "\n";
                }
            }
            continue; 
        }
        
        step4_exteriorFloodFill();
        valid = true; 
    }
    std::cout << "Cascaron generado en " << tries << " intentos.\n";
}

std::string SokobanGenerator::getBoardString() const {
    std::string s = "";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            char c = (char)board[y][x];
            if (c == T_FLOOR) c = ' '; // Convertir a piso vacio normal
            if (c == T_IGNORE) c = ' '; // Por si queda algun interior sin conectar
            s += c;
        }
        s += '\n';
    }
    return s;
}

std::vector<std::vector<char>> SokobanGenerator::getBoard() const {
    std::vector<std::vector<char>> charBoard(height, std::vector<char>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            char c = (char)board[y][x];
            if (c == T_FLOOR) c = ' ';
            if (c == T_IGNORE) c = ' ';
            charBoard[y][x] = c;
        }
    }
    return charBoard;
}

Template SokobanGenerator::createTemplate(const std::vector<std::string>& layout) {
    Template t;
    t.grid.assign(5, std::vector<Tile>(5, T_IGNORE));
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (layout[i][j] == '#') t.grid[i][j] = T_WALL;
            else if (layout[i][j] == '.') t.grid[i][j] = T_FLOOR;
        }
    }
    return t;
}

void SokobanGenerator::loadBaseTemplates() {
    // 17 Templates de Figura 4.3 - Tesis Universidad de Talca
    // Basados exactamente en lo mostrado visualmente
    // ' ' = borde (T_IGNORE - no participa en overlap)
    // '#' = muro (T_WALL)
    // '.' = piso (T_FLOOR)
    // El overlap permite que murallas se conecten con murallas y pisos con pisos
    
    // 1. Cuadro: muro en bordes, piso centro
    baseTemplates.push_back(createTemplate({
        "     ",
        " ### ",
        " #.# ",
        " ### ",
        "     "
    }));
    
    // 2. L invertida: muro superior y derecho, piso abajo-izq
    baseTemplates.push_back(createTemplate({
        "     ",
        " ### ",
        " #.# ",
        " #.. ",
        "     "
    }));
    
    // 3. L vertical: muros arriba, piso abajo
    baseTemplates.push_back(createTemplate({
        "     ",
        " ##. ",
        " #.. ",
        " #.. ",
        "     "
    }));
    
    // 4. Escalera: dos bloques superiores + piso inferior
    baseTemplates.push_back(createTemplate({
        "     ",
        " ##. ",
        " ##. ",
        " ... ",
        "     "
    }));
    
    // 5. L espejo: muros arriba derecha, piso abajo
    baseTemplates.push_back(createTemplate({
        "     ",
        " .## ",
        " .## ",
        " ... ",
        "     "
    }));
    
    // 6. T horizontal: muro arriba-centro, piso alrededor
    baseTemplates.push_back(createTemplate({
        "     ",
        " ### ",
        " .#. ",
        " ... ",
        "     "
    }));
    
    // 7. Cruz vertical: muros arriba-abajo, piso medio
    baseTemplates.push_back(createTemplate({
        "     ",
        " .#. ",
        " ### ",
        " .#. ",
        "     "
    }));
    
    // 8. Cuatro puntos: muros en esquinas, piso centro
    baseTemplates.push_back(createTemplate({
        "     ",
        " #.# ",
        " ... ",
        " #.# ",
        "     "
    }));
    
    // 9. T invertida: muro superior, pisos abajo
    baseTemplates.push_back(createTemplate({
        "     ",
        " #.# ",
        " ### ",
        " ... ",
        "     "
    }));
    
    // 10. Muro simple: pared superior
    baseTemplates.push_back(createTemplate({
        "     ",
        " ### ",
        " ... ",
        " ... ",
        "     "
    }));
    
    // 11. Columna izquierda: muro vertical a la izquierda
    baseTemplates.push_back(createTemplate({
        "     ",
        " #.. ",
        " #.. ",
        " #.. ",
        "     "
    }));
    
    // 12. Piso central: muro en medio, pisos alrededor
    baseTemplates.push_back(createTemplate({
        "     ",
        " ... ",
        " .#. ",
        " ... ",
        "     "
    }));
    
    // 13. Forma compleja: muros en esquinas y centro
    baseTemplates.push_back(createTemplate({
        "     ",
        " #.# ",
        " .#. ",
        " ### ",
        "     "
    }));
    
    // 14. L pequeña: muros arriba-izq, piso abajo
    baseTemplates.push_back(createTemplate({
        "     ",
        " ##. ",
        " #.. ",
        " ... ",
        "     "
    }));
    
    // 15. Forma diagonal: muros dispersos
    baseTemplates.push_back(createTemplate({
        "     ",
        " ##. ",
        " ..# ",
        " .## ",
        "     "
    }));
    
    // 16. Pared central: muro en medio horizontalmente
    baseTemplates.push_back(createTemplate({
        "     ",
        " ... ",
        " ### ",
        " ... ",
        "     "
    }));
    
    // 17. Todo abierto: piso completo (máxima conectividad)
    baseTemplates.push_back(createTemplate({
        "     ",
        " ... ",
        " ... ",
        " ... ",
        "     "
    }));
}

bool SokobanGenerator::step2_placeTemplates() {
    for (int y = 0; y < factorY; ++y) {
        for (int x = 0; x < factorX; ++x) {
            int boardX = x * 3;
            int boardY = y * 3;
            
            std::vector<Template> candidates = getValidCandidates(boardX, boardY);
            if (candidates.empty()) return false; 
            
            std::uniform_int_distribution<int> dist(0, candidates.size() - 1);
            applyTemplate(candidates[dist(rng)], boardX, boardY);
        }
    }
    return true;
}

std::vector<Template> SokobanGenerator::getValidCandidates(int startX, int startY) {
    std::vector<Template> candidates;
    for (const auto& tmpl : baseTemplates) {
        std::vector<Template> variations = generateVariations(tmpl);
        for (const auto& var : variations) {
            if (canPlace(var, startX, startY)) {
                candidates.push_back(var);
            }
        }
    }
    return candidates;
}

bool SokobanGenerator::canPlace(const Template& tmpl, int startX, int startY) {
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            Tile boardTile = board[startY + i][startX + j];
            Tile tmplTile = tmpl.grid[i][j];
            
            // "A non-blank tile must match any pattern it overlaps"
            // T_IGNORE (borde) no restringe nada
            // T_WALL debe coincidir con T_WALL
            // T_FLOOR debe coincidir con T_FLOOR
            if (boardTile != T_IGNORE && tmplTile != T_IGNORE) {
                if (boardTile != tmplTile) return false;
            }
        }
    }
    return true;
}

void SokobanGenerator::applyTemplate(const Template& tmpl, int startX, int startY) {
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            // Solo aplicar tiles que no sean T_IGNORE (bordes)
            if (tmpl.grid[i][j] != T_IGNORE) {
                board[startY + i][startX + j] = tmpl.grid[i][j];
            }
        }
    }
}

bool SokobanGenerator::step3_postProcessing() {
    if (hasLargeSpaces()) {
        // std::cout << "    Falló hasLargeSpaces\n";
        return false;
    }
    if (!isConnected()) {
        // std::cout << "    Falló isConnected\n";
        return false;
    }
    return true;
}

bool SokobanGenerator::hasLargeSpaces() {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (checkBlock(x, y, 4, 3) || checkBlock(x, y, 3, 4)) return true;
        }
    }
    return false;
}

bool SokobanGenerator::checkBlock(int startX, int startY, int w, int h) {
    if (startX + w > width || startY + h > height) return false;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if (board[startY + i][startX + j] != T_FLOOR) return false;
        }
    }
    return true;
}

bool SokobanGenerator::isConnected() {
    int startX = -1, startY = -1;
    int totalFloors = 0;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (board[y][x] == T_FLOOR) {
                if (startX == -1) { startX = x; startY = y; }
                totalFloors++;
            }
        }
    }

    if (totalFloors == 0) return false;

    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
    std::queue<std::pair<int, int>> q;
    
    q.push({startX, startY});
    visited[startY][startX] = true;
    int connectedFloors = 0;

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();
        connectedFloors++;

        for (int i = 0; i < 4; ++i) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (board[ny][nx] == T_FLOOR && !visited[ny][nx]) {
                    visited[ny][nx] = true;
                    q.push({nx, ny});
                }
            }
        }
    }
    return connectedFloors == totalFloors;
}

void SokobanGenerator::step4_exteriorFloodFill() {
    std::queue<std::pair<int, int>> q;
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                if (board[y][x] != T_FLOOR) {
                    q.push({x, y});
                    visited[y][x] = true;
                }
            }
        }
    }

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();

        if (board[cy][cx] == T_IGNORE) board[cy][cx] = T_WALL;

        for (int i = 0; i < 4; ++i) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (!visited[ny][nx] && board[ny][nx] != T_FLOOR) {
                    visited[ny][nx] = true;
                    q.push({nx, ny});
                }
            }
        }
    }
}

Template SokobanGenerator::rotate90(const Template& t) {
    Template res;
    res.grid.assign(5, std::vector<Tile>(5, T_IGNORE));
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            res.grid[j][4 - i] = t.grid[i][j];
        }
    }
    return res;
}

Template SokobanGenerator::flipX(const Template& t) {
    Template res;
    res.grid.assign(5, std::vector<Tile>(5, T_IGNORE));
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            res.grid[i][4 - j] = t.grid[i][j];
        }
    }
    return res;
}

std::vector<Template> SokobanGenerator::generateVariations(const Template& tmpl) {
    std::vector<Template> variations;
    Template current = tmpl;
    
    for (int i = 0; i < 4; ++i) {
        variations.push_back(current);
        variations.push_back(flipX(current));
        current = rotate90(current);
    }
    return variations; 
}