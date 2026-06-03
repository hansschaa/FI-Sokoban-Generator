#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>

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
    // ./evolution_generator ES   <fitness>
    // ./evolution_generator GA   <fitness>
    // ./evolution_generator SA   <fitness>
    //
    // fitness: pushes | expanded | solution_length | branching
    //
    // ./evolution_generator ES pushes

    if (argc < 3)
    {
        std::cout
            << "Usage:\n"
            << "./evolution_generator ES  <fitness>\n"
            << "./evolution_generator GA  <fitness>\n"
            << "./evolution_generator SA  <fitness>\n"
            << "\n"
            << "fitness options:\n"
            << "  pushes           number of box pushes in solution\n"
            << "  expanded         total states generated\n"
            << "  solution_length  number of nodes in solution path\n"
            << "  branching        effective branching factor\n";
        return 1;
    }

    std::string algorithm  = argv[1];
    std::string fitness_arg = argv[2];

    //
    // PARSE FITNESS TYPE
    //

    FitnessType fitnessType;

    if (fitness_arg == "FO1" || fitness_arg == "pushes")
    {
        fitnessType = FitnessType::FO1_PUSHES;
        std::cout << "Fitness: FO1 (Pushes)\n";
    }
    else if (fitness_arg == "FO2" || fitness_arg == "astar_bf")
    {
        fitnessType = FitnessType::FO2_ASTAR_EFF_BF;
        std::cout << "Fitness: FO2 (A* Effective Branching Factor)\n";
    }
    else if (fitness_arg == "FO3" || fitness_arg == "sol_bf")
    {
        fitnessType = FitnessType::FO3_SOL_EFF_BF;
        std::cout << "Fitness: FO3 (Solution Effective Branching Factor)\n";
    }
    else if (fitness_arg == "FO4" || fitness_arg == "deadlocks")
    {
        fitnessType = FitnessType::FO4_DEADLOCKS;
        std::cout << "Fitness: FO4 (Deadlocks)\n";
    }
    else
    {
        std::cerr
            << "Unknown fitness: \"" << fitness_arg << "\"\n"
            << "Valid options: FO1 | FO2 | FO3 | FO4\n";
        return 1;
    }

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
    // max_boxes: upper bound available to mutation operators that add boxes.
    // The initial population always starts with 1 box; complexity grows
    // through evolution, not initialization.
    //

    const int free_cells = count_free_cells(shell);
    const int max_boxes  = std::max(1, std::min(6, free_cells / 15));

    std::cout
        << "\nBOARD SHELL:\n"
        << board_to_pretty_string(shell)
        << "Free cells: " << free_cells
        << "  |  Max boxes for mutation: " << max_boxes << "\n\n";

    //
    // INITIAL POPULATION
    //

    std::vector<Individual> population;

    Evaluator evaluator;
    evaluator.fitnessType = fitnessType;

    const int POP_SIZE = 10;

    //
    // GENERATE VALID INDIVIDUALS
    //

    for (int i = 0; i < POP_SIZE; i++)
    {
        bool valid    = false;
        int attempts  = 0;

        while (!valid && attempts < 10000)
        {
            auto board = shell;

            //
            // All initial individuals start with 1 box.
            // Complexity grows through evolution, not initialization.
            //

            int numBoxes = 1;

            placeRandom(board, '@');

            for (int k = 0; k < numBoxes; k++)
            {
                placeRandom(board, '$');
                placeRandom(board, '.');
            }

            std::string  level = board_to_string(board);
            unsigned int rows  = board.size();
            unsigned int cols  = board[0].size();

            game_solver solver(level, rows, cols, 512);

            std::vector<game_node> solution;

            auto stats = solver.test_template(Method::a_star, solution);

            if (stats.status == SolveStatus::SOLVED)
            {
                Individual ind;
                ind.board   = board;
                ind.fitness = evaluator.evaluate(ind);

                population.push_back(ind);
                valid = true;

                std::cout
                    << "VALID INDIVIDUAL " << i
                    << "  |  boxes=" << numBoxes
                    << "  |  fitness=" << ind.fitness << "\n"
                    << board_to_pretty_string(board) << "\n";
            }

            attempts++;
        }

        if (!valid)
        {
            std::cerr << "Could not generate valid individual " << i << "\n";
        }
    }

    //
    // POPULATION SAFETY CHECK
    //

    if (population.empty())
    {
        std::cerr << "ERROR: could not generate any valid individual. Aborting.\n";
        return 1;
    }

    std::cout << "POPULATION READY: " << population.size() << " individuals\n\n";

    //
    // FINAL BEST
    //

    Individual best;

    //
    // RUN ALGORITHM
    //

    if (algorithm == "ES")
    {
        EvolutionStrategy es;
        es.mu              = 5;
        es.lambda          = 10;  // igualado a GA para comparación justa
        es.maxEvaluations  = 500;
        es.stagnationLimit = 5;

        std::cout << "RUNNING MU + LAMBDA ES\n";
        best = es.run(population);
    }
    else if (algorithm == "GA")
    {
        GeneticAlgorithm ga;
        ga.offspringSize   = 10;
        ga.maxEvaluations  = 500;
        ga.stagnationLimit = 15;

        std::cout << "RUNNING GENETIC ALGORITHM\n";
        best = ga.run(population);
    }
    else if (algorithm == "SA")
    {
        SimulatedAnnealing sa;
        sa.initialTemperature = 100.0;
        sa.coolingRate        = 0.01;
        sa.maxEvaluations     = 500;
        sa.stagnationLimit    = 15;

        //
        // START FROM BEST INDIVIDUAL
        // Avoids dependency on random ordering of the initial population.
        //
        Individual initial = *std::max_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });

        std::cout << "RUNNING SIMULATED ANNEALING\n";
        best = sa.run(initial);
    }
    else
    {
        std::cerr << "Unknown algorithm: " << algorithm << "\nUse ES, GA or SA\n";
        return 1;
    }

    //
    // FINAL RESULT
    //

    std::cout
        << "\n========================\n"
        << "FINAL BEST FITNESS = " << best.fitness << "\n"
        << "\nBEST BOARD:\n"
        << board_to_pretty_string(best.board) << "\n";


    //
    // ========================================================
    // ANÁLISIS PROFUNDO DEL MEJOR INDIVIDUO (CAMPEÓN EVOLUTIVO)
    // ========================================================
    //

    std::cout << "\nIniciando analisis profundo del campeon...\n";

    std::string champion_level = board_to_string(best.board);
    unsigned int champ_rows = best.board.size();
    unsigned int champ_cols = best.board.empty() ? 0 : best.board[0].size();

    // Crear un solver exclusivamente para el campeón
    game_solver champ_solver(champion_level, champ_rows, champ_cols, 512);
    std::vector<game_node> champ_solution;

    // Ejecutar A* con Heurística Húngara y activando el simulador de Path (Flag = true)
    auto champ_stats = champ_solver.test_template(Method::a_star, Heuristic::hungarian, champ_solution, true);

    std::cout << "\n=========================================\n";
    std::cout << "  DUMP COMPLETO DE STATS (TABLERO FINAL) \n";
    std::cout << "=========================================\n";

    // AÑADIDO: Imprimir el mapa visual del mejor individuo
    std::cout << "[TABLERO GENERADO]\n";
    std::cout << board_to_pretty_string(best.board) << "\n";

    print_solver_stats(champ_stats);

    return 0;
}