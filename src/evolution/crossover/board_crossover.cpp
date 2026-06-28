#include "../../../include/evolution/crossover/board_crossover.h"

#include "../../../include/evolution/utils/board_utils.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// COUNT HELPERS
// ─────────────────────────────────────────────────────────────────────────────

int BoardCrossover::countBoxes(
    const std::vector<std::vector<char>>& board)
{
    int count = 0;
    for (const auto& row : board)
        for (char c : row)
            if (c == '$' || c == '*') count++;
    return count;
}

int BoardCrossover::countGoals(
    const std::vector<std::vector<char>>& board)
{
    int count = 0;
    for (const auto& row : board)
        for (char c : row)
            if (c == '.' || c == '+' || c == '*') count++;
    return count;
}

int BoardCrossover::countPlayers(
    const std::vector<std::vector<char>>& board)
{
    int count = 0;
    for (const auto& row : board)
        for (char c : row)
            if (c == '@' || c == '+') count++;
    return count;
}

// ─────────────────────────────────────────────────────────────────────────────
// GET RANDOM EMPTY CELL
// ─────────────────────────────────────────────────────────────────────────────

Pair BoardCrossover::getRandomEmpty(
    const std::vector<std::vector<char>>& board)
{
    std::vector<Pair> empty;

    for (int i = 0; i < (int)board.size(); i++)
        for (int j = 0; j < (int)board[i].size(); j++)
            if (board[i][j] == ' ')
                empty.push_back({i, j});

    if (empty.empty())
        return {-1, -1};

    return empty[rand() % empty.size()];
}

// ─────────────────────────────────────────────────────────────────────────────
// IS STRUCTURALLY VALID
// boxes == goals > 0 and exactly 1 player
// Does NOT call the solver
// ─────────────────────────────────────────────────────────────────────────────

bool BoardCrossover::isStructurallyValid(
    const std::vector<std::vector<char>>& board)
{
    int boxes   = countBoxes(board);
    int goals   = countGoals(board);
    int players = countPlayers(board);

    return boxes == goals &&
           boxes  > 0     &&
           players == 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// GET INTERESTING REGIONS
// Horizontal and vertical regions of length crossoverSpacing
// that contain at least one game element and no walls
// ─────────────────────────────────────────────────────────────────────────────

std::vector<BoardCrossover::CrossPair> BoardCrossover::getInterestingRegions(
    const std::vector<std::vector<char>>& board)
{
    std::vector<CrossPair> regions;

    int rows = board.size();
    int cols = board[0].size();

    Pair directions[2] = {{0, 1}, {1, 0}};

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (board[i][j] == '#')
                continue;

            for (const auto& dir : directions)
            {
                int endR = i + dir.i * (crossoverSpacing - 1);
                int endC = j + dir.j * (crossoverSpacing - 1);

                if (endR >= rows || endC >= cols)
                    continue;

                bool hasGameElement = false;
                bool hasWall        = false;

                for (int k = 0; k < crossoverSpacing; k++)
                {
                    char ch = board[i + dir.i * k][j + dir.j * k];

                    if (ch == '#')          { hasWall = true; break; }
                    if (ch == '@' || ch == '+' ||
                        ch == '$' || ch == '*' ||
                        ch == '.')            hasGameElement = true;
                }

                if (hasGameElement && !hasWall)
                    regions.push_back({{i, j}, dir});
            }
        }
    }

    return regions;
}

// ─────────────────────────────────────────────────────────────────────────────
// APPLY REGION
// Copies a region from source into board at the same coordinates
// ─────────────────────────────────────────────────────────────────────────────

void BoardCrossover::applyRegion(
    std::vector<std::vector<char>>& board,
    const std::vector<std::vector<char>>& source,
    const CrossPair& region)
{
    for (int k = 0; k < crossoverSpacing; k++)
    {
        int r = region.pivot.i + region.direction.i * k;
        int c = region.pivot.j + region.direction.j * k;
        board[r][c] = source[r][c];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// REPAIR ILLEGAL
// Fixes player count, box/goal imbalance, enforces dynamic maxBoxes
// Does NOT call the solver
// ─────────────────────────────────────────────────────────────────────────────

void BoardCrossover::repairIllegal(
    std::vector<std::vector<char>>& board)
{
    //
    // FIX PLAYER COUNT
    //

    int players = countPlayers(board);

    if (players == 0)
    {
        Pair e = getRandomEmpty(board);

        if (e.i != -1)
            board[e.i][e.j] = '@';
    }
    else if (players > 1)
    {
        int removed = 0;

        for (int i = 0;
             i < (int)board.size() && removed < players - 1;
             i++)
        {
            for (int j = 0;
                 j < (int)board[i].size() && removed < players - 1;
                 j++)
            {
                if (board[i][j] == '@')
                {
                    board[i][j] = ' ';
                    removed++;
                }
                else if (board[i][j] == '+')
                {
                    board[i][j] = '.';
                    removed++;
                }
            }
        }
    }

    //
    // DYNAMIC MAX BOXES
    // 1 box per 15 free cells, clamped to [3, 6].
    // Minimum of 3 ensures small shells aren't trivially easy.
    // Maximum of 6 prevents overcrowding on large shells.
    // Calculated from the current board state so it adapts
    // as the board fills up during evolution.
    //

    const int free_cells = count_free_cells(board);
    const int maxBoxes   = 7;

    //
    // ENFORCE MAX BOXES
    //

    int boxes = countBoxes(board);
    int goals = countGoals(board);

    while (boxes > maxBoxes)
    {
        for (int i = 0;
             i < (int)board.size() && boxes > maxBoxes;
             i++)
        {
            for (int j = 0;
                 j < (int)board[i].size() && boxes > maxBoxes;
                 j++)
            {
                if (board[i][j] == '$')
                {
                    board[i][j] = ' ';
                    boxes--;
                }
                else if (board[i][j] == '*')
                {
                    board[i][j] = '.';
                    boxes--;
                }
            }
        }
    }

    goals = countGoals(board);

    while (goals > maxBoxes)
    {
        for (int i = 0;
             i < (int)board.size() && goals > maxBoxes;
             i++)
        {
            for (int j = 0;
                 j < (int)board[i].size() && goals > maxBoxes;
                 j++)
            {
                if (board[i][j] == '.')
                {
                    board[i][j] = ' ';
                    goals--;
                }
                else if (board[i][j] == '+')
                {
                    board[i][j] = '@';
                    goals--;
                }
            }
        }
    }

    //
    // FIX BOX / GOAL IMBALANCE
    //

    boxes = countBoxes(board);
    goals = countGoals(board);

    while (boxes > goals)
    {
        Pair e = getRandomEmpty(board);

        if (e.i == -1) break;

        board[e.i][e.j] = '.';
        goals++;
    }

    while (goals > boxes)
    {
        Pair e = getRandomEmpty(board);

        if (e.i == -1) break;

        board[e.i][e.j] = '$';
        boxes++;
    }

    //
    // ENSURE AT LEAST ONE BOX AND ONE GOAL
    //

    boxes = countBoxes(board);
    goals = countGoals(board);

    if (boxes == 0)
    {
        Pair e1 = getRandomEmpty(board);

        if (e1.i == -1) return;

        std::vector<std::vector<char>> temp = board;
        temp[e1.i][e1.j] = '$';

        Pair e2 = getRandomEmpty(temp);

        if (e2.i == -1) return;

        board[e1.i][e1.j] = '$';
        board[e2.i][e2.j] = '.';
    }

    if (goals == 0)
    {
        Pair e1 = getRandomEmpty(board);

        if (e1.i == -1) return;

        std::vector<std::vector<char>> temp = board;
        temp[e1.i][e1.j] = '.';

        Pair e2 = getRandomEmpty(temp);

        if (e2.i == -1) return;

        board[e1.i][e1.j] = '.';
        board[e2.i][e2.j] = '$';
    }

    //
    // ENSURE EXACTLY ONE PLAYER
    //

    players = countPlayers(board);

    if (players == 0)
    {
        Pair e = getRandomEmpty(board);

        if (e.i != -1)
            board[e.i][e.j] = '@';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// APPLY
// Main crossover operator
// Solver is NOT called here — evaluator handles solvability
// ─────────────────────────────────────────────────────────────────────────────

bool BoardCrossover::apply(
    const Individual& parent1,
    const Individual& parent2,
    Individual& child)
{
    //
    // CHILD STARTS AS COPY OF PARENT1
    //

    child = parent1;

    //
    // GET INTERESTING REGIONS FROM PARENT2
    //

    auto regions = getInterestingRegions(parent2.board);

    if (regions.empty())
    {
        //std::cout << "CROSSOVER: NO INTERESTING REGIONS\n";
        return false;
    }

    //
    // SELECT AND APPLY RANDOM REGION
    //

    const CrossPair& selected = regions[rand() % regions.size()];

    applyRegion(child.board, parent2.board, selected);

    //
    // IF STRUCTURALLY VALID → DONE
    //

    if (isStructurallyValid(child.board))
    {
        //std::cout << "CROSSOVER: SUCCESS\n";
        return true;
    }

    //
    // ATTEMPT REPAIR
    //

    //std::cout << "CROSSOVER: ATTEMPTING REPAIR\n";

    repairIllegal(child.board);

    if (isStructurallyValid(child.board))
    {
        //std::cout << "CROSSOVER: REPAIR SUCCESS\n";
        return true;
    }

    //std::cout << "CROSSOVER: REPAIR FAILED\n";
    return false;
}