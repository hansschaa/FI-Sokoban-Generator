#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>
#include <future>
#include <mutex>
#include <chrono>
#include <atomic>

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
    // ./evolution_generator ES   <fitness> [runs]
    // ./evolution_generator GA   <fitness> [runs]
    // ./evolution_generator SA   <fitness> [runs]
    //
    // fitness: pushes | expanded | solution_length | branching
    //
    // ./evolution_generator ES pushes 5

    if (argc < 3)
    {
        std::cout
            << "Usage:\n"
            << "./evolution_generator ES  <fitness> [runs] [--show-stats] [--no-parallel]\n"
            << "./evolution_generator GA  <fitness> [runs] [--show-stats] [--no-parallel]\n"
            << "./evolution_generator SA  <fitness> [runs] [--show-stats] [--no-parallel]\n"
            << "\n"
            << "fitness options:\n"
            << "  pushes           number of box pushes in solution\n"
            << "  expanded         total states generated\n"
            << "  solution_length  number of nodes in solution path\n"
            << "  branching        effective branching factor\n";
        return 1;
    }

    bool show_stats = false;
    bool no_parallel = false;
    int num_runs = 1;
    
    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--show-stats") {
            show_stats = true;
        } else if (arg == "--no-parallel") {
            no_parallel = true;
        } else {
            // If it's a number, it's the runs argument
            try {
                num_runs = std::stoi(arg);
            } catch (...) {}
        }
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
    else if (fitness_arg == "FO5" || fitness_arg == "repeated_nodes")
    {
        fitnessType = FitnessType::FO5_REPEATED_NODES;
        std::cout << "Fitness: FO5 (Repeated Nodes)\n";
    }
    else
    {
        std::cerr
            << "Unknown fitness: \"" << fitness_arg << "\"\n"
            << "Valid options: FO1 | FO2 | FO3 | FO4 | FO5\n";
        return 1;
    }

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) hardware_threads = 4;
    std::cout << "Available Hardware Threads: " << hardware_threads << "\n";
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
    // COMPUTE DEADLOCK MASK
    //
    auto deadlock_mask = compute_deadlock_mask(shell);
    int deadlocks_count = 0;
    for (const auto& row : deadlock_mask) {
        for (bool b : row) if (b) deadlocks_count++;
    }
    std::cout << "Computed deadlock mask. Deadlock cells found: " << deadlocks_count << "\n\n";

    Evaluator evaluator;
    evaluator.fitnessType = fitnessType;

    const int POP_SIZE = 10;

    for (int run = 0; run < num_runs; run++)
    {
        std::cout << "\n=========================================\n";
        std::cout << "  RUN " << (run + 1) << " / " << num_runs << "\n";
        std::cout << "=========================================\n\n";

        //
        // INITIAL POPULATION
        //

        std::vector<Individual> population;

        //
        // GENERATE VALID INDIVIDUALS
        //

        std::mutex pop_mutex;
        
        auto start_time = std::chrono::high_resolution_clock::now();

        auto generate_individual = [&](int i) {
            bool valid = false;
            int attempts = 0;

            while (!valid && attempts < 10000)
            {
                auto board = shell;
                int numBoxes = 1;

                // Bloqueamos rand() y la generacion por seguridad de hilos
                {
                    std::lock_guard<std::mutex> lock(pop_mutex);
                    placeRandom(board, '@', deadlock_mask);
                    for (int k = 0; k < numBoxes; k++)
                    {
                        placeRandom(board, '$', deadlock_mask);
                        placeRandom(board, '.', deadlock_mask);
                    }
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
                    
                    std::lock_guard<std::mutex> lock(pop_mutex);
                    ind.fitness = evaluator.evaluate(ind);

                    population.push_back(ind);
                    valid = true;

                    std::cout
                        << "VALID INDIVIDUAL " << i
                        << "  |  boxes=" << numBoxes
                        << "  |  fitness=" << ind.fitness << "\n";
                }

                attempts++;
            }

            if (!valid)
            {
                std::lock_guard<std::mutex> lock(pop_mutex);
                std::cerr << "Could not generate valid individual " << i << "\n";
            }
        };

        if (no_parallel) {
            for (int i = 0; i < POP_SIZE; i++) {
                generate_individual(i);
            }
        } else {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4; // Fallback
            
            unsigned int threads_to_launch = std::min((unsigned int)POP_SIZE, num_threads);

            std::atomic<int> current_task{0};
            std::vector<std::future<void>> futures;

            auto worker_task = [&]() {
                while (true) {
                    int i = current_task.fetch_add(1);
                    if (i >= POP_SIZE) {
                        break; // No more individuals to generate
                    }
                    generate_individual(i);
                }
            };

            for (unsigned int t = 0; t < threads_to_launch; t++) {
                futures.push_back(std::async(std::launch::async, worker_task));
            }
            
            for (auto& f : futures) {
                f.get();
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    //
    // POPULATION SAFETY CHECK
    //

    if (population.empty())
    {
        std::cerr << "ERROR: could not generate any valid individual. Aborting.\n";
        return 1;
    }

    std::cout << "POPULATION READY: " << population.size() << " individuals\n";
    std::cout << "Generation Time: " << duration.count() << " ms\n\n";

    //
    // FINAL BEST
    //

    Individual best;

    //
    // RUN ALGORITHM
    //
    
    auto t_start_run = std::chrono::high_resolution_clock::now();

    if (algorithm == "ES")
    {
        EvolutionStrategy es;
        es.use_parallel    = !no_parallel;
        es.setDeadlockMask(deadlock_mask);
        es.mu              = 10;
        es.lambda          = 20;  // igualado a GA para comparación justa
        es.maxEvaluations  = 20000;
        es.stagnationLimit = 100;

        std::cout << "RUNNING MU + LAMBDA ES\n";
        best = es.run(population);
    }
    else if (algorithm == "GA")
    {
        GeneticAlgorithm ga;
        ga.use_parallel    = !no_parallel;
        ga.setDeadlockMask(deadlock_mask);
        ga.offspringSize   = 10;
        ga.maxEvaluations  = 2000;
        ga.stagnationLimit = 30;

        std::cout << "RUNNING GENETIC ALGORITHM\n";
        best = ga.run(population);
    }
    else if (algorithm == "SA")
    {
        SimulatedAnnealing sa;
        sa.setDeadlockMask(deadlock_mask);
        sa.initialTemperature = 100.0;
        sa.coolingRate        = 0.01;
        sa.maxEvaluations     = 500;
        sa.stagnationLimit    = 15;

        //
        // START FROM BEST INDIVIDUAL
        // Avoids dependency on random ordering of the initial population.
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

    auto t_end_run = std::chrono::high_resolution_clock::now();
    long long run_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_run - t_start_run).count();
    std::cout << "Total Evolution Time: " << run_time << " ms\n";

    //
    // FINAL RESULT OF THIS RUN
    //

    std::cout
        << "\n========================\n"
        << "BEST FITNESS FOR RUN " << (run + 1) << " = " << best.fitness << "\n"
        << "\nBEST BOARD:\n"
        << board_to_pretty_string(best.board) << "\n";


    //
    // ========================================================
    // ANÁLISIS PROFUNDO DEL MEJOR INDIVIDUO (CAMPEÓN EVOLUTIVO)
    // ========================================================
    //

    std::cout << "\nIniciando analisis profundo del campeon de la corrida " << (run + 1) << "...\n";

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

    if (show_stats) {
        print_solver_stats(champ_stats);
    }

    } // END OF RUN LOOP

    return 0;
}