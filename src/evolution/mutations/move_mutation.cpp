#include "../../../include/evolution/mutations/move_mutation.h"

#include "../../../include/evolution/utils/board_utils.h"

#include <cstdlib>

void MoveMutation::apply(Individual& ind)
{
    auto& b = ind.board;

    std::vector<Pair> elements;
    std::vector<Pair> empty;

    for (int i = 0; i < b.size(); i++) {

        for (int j = 0; j < b[i].size(); j++) {

            char c = b[i][j];

            if (c == '@' || c == '$' || c == '.')
                elements.push_back({i,j});

            if (c == ' ')
                empty.push_back({i,j});
        }
    }

    if (elements.empty() || empty.empty())
        return;

    Pair from =
        elements[rand() % elements.size()];

    Pair to =
        empty[rand() % empty.size()];

    b[to.i][to.j] =
        b[from.i][from.j];

    b[from.i][from.j] = ' ';
}