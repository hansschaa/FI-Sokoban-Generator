#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "game_solver.h"

using namespace std;

void read_file(const string& filename, int& mm, int& nn, string& temp) {
    mm = 0;
    nn = 0;
    temp.clear();

    ifstream file_read(filename);

    if (!file_read) {
        cerr << "cannot open file: " << filename << endl;
        exit(1);
    }

    string line;
    vector<string> lines;

    while(getline(file_read, line)) {
        if(!line.empty() && line.back() == '\r')
            line.pop_back();

        lines.push_back(line);
    }

    mm = lines.size();

    int maxw = 0;
    for(auto& l : lines)
        maxw = max(maxw, (int)l.size());

    nn = maxw;

    for(auto& l : lines) {
        while((int)l.size() < nn)
            l.push_back('#');

        temp += l;
    }
}

int main(int argc, char** argv) {

    if(argc < 4) {
        cerr << "usage:\n";
        cerr << "./solver_benchmark board.txt algorithm memoryMB\n";
        cerr << "algorithm: 0=A* 1=DFS 2=BFS\n";
        return 1;
    }

    string boardfile = argv[1];
    int alg = stoi(argv[2]);
    int mem = stoi(argv[3]);

    int mm, nn;
    string temp;

    read_file(boardfile, mm, nn, temp);

    game_solver ga(temp, mm, nn, mem);

    auto start = chrono::high_resolution_clock::now();

    std::vector<game_node> solution;

    SolverStats stats =
        ga.test_template(int_to_method(alg), solution);

    auto end = chrono::high_resolution_clock::now();

    cout << "board=" << boardfile << endl;
    cout << "algorithm=" << alg << endl;
    cout << "runtime_ms=" << stats.runtime_ms << endl;
    cout << "solution_length=" << stats.pushes << endl;
    cout << "explored_states=" << stats.generated_states << endl;
    cout << "status=";

    if(stats.status == SolveStatus::SOLVED)
        cout << "SOLVED";
    else
        cout << "UNSOLVABLE";

    cout << endl;

    return 0;
}