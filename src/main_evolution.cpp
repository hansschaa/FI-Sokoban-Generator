#include "../include/game_solver.h"
#include "../include/game_node.h"

#include "../include/evolution/algorithms/evolution_strategy.h"
#include "../include/evolution/evaluator.h"

#include <iostream>

int main()
{
    std::cout << "START\n";

    //
    // TABLERO INICIAL VÁLIDO
    //

    Individual ind;

    ind.board = {

        {'#','#','#','#','#','#','#'},
        {'#',' ',' ',' ',' ',' ','#'},
        {'#',' ','$','.',' ',' ','#'},
        {'#',' ',' ','@',' ',' ','#'},
        {'#','#','#','#','#','#','#'}
    };

    //
    // DEBUG: imprimir tablero
    //

    std::cout << "INITIAL BOARD:\n";

    for (auto& row : ind.board) {

        for (char c : row)
            std::cout << c;

        std::cout << '\n';
    }

    std::cout << std::endl;

    //
    // PRIMERO probar SOLO evaluator
    // SIN metaheurística todavía
    //

    Evaluator eval;

    std::cout << "CALLING EVALUATOR\n";

    double fit =
        eval.evaluate(ind);

    std::cout << "FITNESS = "
              << fit
              << std::endl;

    //
    // SOLO si funciona el evaluator,
    // entonces correr ES
    //

    if (fit > -1e8) {

        std::cout << "\nRUNNING EVOLUTION STRATEGY\n";

        EvolutionStrategy es;

        Individual best =
            es.run(ind);

        std::cout << "\nBEST FITNESS = "
                  << best.fitness
                  << std::endl;

        std::cout << "\nBEST BOARD:\n";

        for (auto& row : best.board) {

            for (char c : row)
                std::cout << c;

            std::cout << '\n';
        }
    }
    else {

        std::cout << "\nINVALID INITIAL BOARD\n";
    }

    return 0;
}