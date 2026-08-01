#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream file("../src/batch_solver.cpp");
    std::string line;
    while(std::getline(file, line)) {
        if(line.find("continue") != std::string::npos || line.find("break") != std::string::npos) {
            std::cout << line << std::endl;
        }
    }
    return 0;
}
