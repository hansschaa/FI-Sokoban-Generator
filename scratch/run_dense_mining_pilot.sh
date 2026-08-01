#!/bin/bash
set -e

echo "=========================================================="
echo " PILOT: Generating Dense Solvable Boards (ES)"
echo "=========================================================="
mkdir -p training_data/DenseSolvablesPilot
# Use 20 runs for a quick pilot
./build/dense_dataset_miner --runs 20
