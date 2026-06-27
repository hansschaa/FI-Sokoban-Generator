#include <iostream>
#include <string>
#include <cstdlib>
#include "../include/shell_generator/shell_generator.h"

int main(int argc, char** argv) {
    int factorX = 2;
    int factorY = 2;
    int count = 1;

    if (argc >= 3) {
        factorX = std::stoi(argv[1]);
        factorY = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        count = std::stoi(argv[3]);
    }

    std::cout << "Generando " << count << " cascaron(es) con factor " << factorX << "x" << factorY << "...\n\n";

    for (int i = 0; i < count; i++) {
        std::cout << "===== CASCARON " << (i + 1) << " =====\n";
        SokobanGenerator generator(factorX, factorY);
        generator.generate();
        std::string board = generator.getBoardString();
        
        std::cout << board << "\n";
    }

    return 0;
}
