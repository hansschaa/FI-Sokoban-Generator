#ifndef SHELLGENERATOR_H
#define SHELLGENERATOR_H

#include <vector>
#include <string>
#include <random>

// Definición de los tipos de casillas
enum Tile { T_IGNORE = ' ', T_WALL = '#', T_FLOOR = '.' };

// Estructura para manejar los templates de 5x5
struct Template {
    std::vector<std::vector<Tile>> grid;
};

class SokobanGenerator {
private:
    int factorX, factorY;
    int width, height;
    std::vector<std::vector<Tile>> board;
    std::vector<Template> baseTemplates;
    std::mt19937 rng;

    // Métodos privados (Lógica interna)
    Template createTemplate(const std::vector<std::string>& layout);
    void loadBaseTemplates();
    bool step2_placeTemplates();
    std::vector<Template> getValidCandidates(int startX, int startY);
    bool canPlace(const Template& tmpl, int startX, int startY);
    void applyTemplate(const Template& tmpl, int startX, int startY);
    bool step3_postProcessing();
    bool hasLargeSpaces();
    bool checkBlock(int startX, int startY, int w, int h);
    bool isConnected();
    void step4_exteriorFloodFill();
    Template rotate90(const Template& t);
    Template flipX(const Template& t);
    std::vector<Template> generateVariations(const Template& tmpl);

public:
    // Constructor: recibe los factores de tamaño
    SokobanGenerator(int fX, int fY);
    
    // Método principal para generar un nivel válido
    void generate();
    
    // Retorna el cascarón generado como un string
    std::string getBoardString() const;

    // Retorna el cascarón como matriz de caracteres
    std::vector<std::vector<char>> getBoard() const;
};

#endif // SHELLGENERATOR_H