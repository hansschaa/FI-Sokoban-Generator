#include <iostream>
#include <vector>
#include <ctime>
#include <string>

#include "../include/evolution/algorithms/evolution_strategy.h"
#include "../include/evolution/algorithms/genetic_algorithm.h"

#include "../include/evolution/individual.h"
#include "../include/evolution/utils/board_utils.h"

#include "../include/game_solver.h"

#include "../include/evolution/algorithms/simulated_annealing.h"

int main(int argc, char** argv)
{
    srand(time(nullptr));

    //
    // ARGUMENT CHECK
    //
    // Usage:
    // ./evolution_generator ES
    // ./evolution_generator GA
    //

    if (argc < 2)
    {
        std::cout
            << "Usage:\n"
            << "./evolution_generator ES\n"
            << "./evolution_generator GA\n";

        return 1;
    }

    std::string algorithm =
        argv[1];

    std::cout << "START\n";

    //
    // BOARD SHELL
    // ONLY WALLS + EMPTY SPACES
    //

    std::vector<std::vector<char>> shell =
    {
        {'#','#','#','#','#','#','#'},
        {'#',' ',' ',' ',' ',' ','#'},
        {'#',' ',' ',' ',' ',' ','#'},
        {'#',' ',' ',' ',' ',' ','#'},
        {'#','#','#','#','#','#','#'}
    };

    //
    // SHOW SHELL
    //

    std::cout
        << "\nBOARD SHELL:\n";

    std::cout
        << board_to_pretty_string(shell)
        << std::endl;

    //
    // INITIAL POPULATION
    //

    std::vector<Individual> population;

    const int POP_SIZE = 10;

    //
    // GENERATE VALID INDIVIDUALS
    //

    for (int i = 0; i < POP_SIZE; i++)
    {
        bool valid = false;

        while (!valid)
        {
            //
            // COPY SHELL
            //

            auto board = shell;

            //
            // PLACE ELEMENTS
            //

            placeRandom(board, '@');

            placeRandom(board, '$');

            placeRandom(board, '.');

            //
            // CONVERT TO STRING
            // IMPORTANT:
            // NO NEWLINES
            //

            std::string level =
                board_to_string(board);

            //
            // SOLVER DIMENSIONS
            //

            unsigned int rows =
                board.size();

            unsigned int cols =
                board[0].size();

            //
            // CREATE SOLVER
            //

            game_solver solver(
                level,
                rows,
                cols,
                512);

            //
            // SOLVE
            //

            std::vector<game_node> solution;

            auto stats =
                solver.test_template(
                    1,
                    solution);

            //
            // ACCEPT ONLY SOLVABLE
            //

            if (stats.status ==
                SolveStatus::SOLVED)
            {
                Individual ind;

                ind.board = board;

                //
                // INITIAL FITNESS
                //

                ind.fitness =
                    stats.pushes;

                population.push_back(ind);

                valid = true;

                std::cout
                    << "\nVALID INDIVIDUAL "
                    << i
                    << "\n";

                std::cout
                    << "FITNESS = "
                    << ind.fitness
                    << "\n";

                std::cout
                    << board_to_pretty_string(board)
                    << std::endl;
            }
        }
    }

    //
    // FINAL BEST
    //

    Individual best;

    //
    // RUN ES
    //

    if (algorithm == "ES")
    {
        EvolutionStrategy es;

        es.mu = 5;

        es.lambda = 7;

        es.maxEvaluations = 500;

        es.stagnationLimit = 15;

        std::cout
            << "\nRUNNING MU + LAMBDA ES\n";

        best =
            es.run(population);
    }

    //
    // RUN GA
    //

    else if (algorithm == "GA")
    {
        GeneticAlgorithm ga;

        ga.offspringSize = 10;

        ga.maxEvaluations = 500;

        ga.stagnationLimit = 15;

        std::cout
            << "\nRUNNING GENETIC ALGORITHM\n";

        best =
            ga.run(population);
    }

    else if (algorithm == "SA")
    {
        SimulatedAnnealing sa;

        sa.initialTemperature = 100.0;

        sa.coolingRate = 0.01;

        sa.maxEvaluations = 500;

        //
        // MUTATION OPERATOR
        //

        MoveMutation moveMutation;

        sa.mutation =
            &moveMutation;

        //
        // INITIAL SOLUTION
        //

        Individual initial =
            population[0];

        std::cout
            << "\nRUNNING SIMULATED ANNEALING\n";

        best =
            sa.run(initial);
    }

    //
    // INVALID ARGUMENT
    //

    else
    {
        std::cout
            << "Unknown algorithm: "
            << algorithm
            << "\n";

        std::cout
            << "Use ES or GA\n";

        return 1;
    }

    //
    // FINAL RESULT
    //

    std::cout
        << "\n========================\n";

    std::cout
        << "FINAL BEST FITNESS = "
        << best.fitness
        << std::endl;

    std::cout
        << "\nBEST BOARD:\n";

    std::cout
        << board_to_pretty_string(best.board)
        << std::endl;

    return 0;
}