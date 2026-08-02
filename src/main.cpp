#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include "game_solver.h"
#include "draw.h"

using namespace std;

void read_file(int& mm, int& nn, string& temp, const string& filename){
    mm=0;
    nn=0;
    temp.clear();
    ifstream file_read;
    file_read.open(filename, ios::in);

    if (!file_read) {
        printf("%s dose not exist!\n", filename.c_str());
        exit(100);
    }

    string x;
    string tempr;
    while (getline(file_read,x)) {
        tempr += x;
        mm += 1;
    }
    for(auto &tc: tempr){
        if(tc != '\r' && tc != '\n'){
            temp.push_back(tc);
        }
    }
    
    if (mm == 0) mm = 1;
    nn = temp.size() / mm;
    file_read.close();
}

int main(int argc, char** argv) {
    int mm;
    int nn;
    string temp;
    if (argc >= 2) {
        read_file(mm, nn, temp, argv[1]);
    } else {
        read_file(mm, nn, temp, "box.txt");
    }
    
    char input = '0';
    int memval = 1000;
    
    if (argc >= 4) {
        input = argv[2][0];
        memval = stoi(argv[3]);
    } else {
        printf("please select your algorithm(enter 0 or 1 or 2)\n");
        printf("0: A*;    1: dfs    2: bfs\n");
        std::cin >> input;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    int iinput = 0;
    if (input == '0' || input == '1' || input == '2') {
        iinput = input - '0';
    }
    else {
        printf("wrong input!!\n");
        exit(-1);
    }

    if (argc < 4) {
        printf("please input the memory you want to use(unit: MB)\n");
        std::cin >> memval;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    game_solver ga(temp, mm, nn, memval);
    std::vector<game_node> solution;

    ga.enable_advanced_deadlocks = true;
    auto stats = ga.test_template(int_to_method(iinput), Heuristic::hungarian, solution);
    printf("Pushes: %d\n", stats.pushes);

    if (argc < 4) {
        printf("press Enter to show solves\n");
        cin.get();
        draw_picture d;
        d.draw(solution);
    }
}