#!/bin/bash
set -e

echo "=========================================================="
echo " STEP 1: Generating Dense Solvable Boards (ES)"
echo "=========================================================="
mkdir -p training_data/DenseSolvables
./build/dense_dataset_miner --runs 2000

echo "=========================================================="
echo " STEP 2: Generating Dense Contrastive Pairs"
echo "=========================================================="
mkdir -p training_data/DenseContrastivePairs
# The contrastive_pair_miner usage is: ./contrastive_pair_miner <solvables_dir> <output_dir> <num_pairs>
# Let's target 100,000 pairs
./build/contrastive_pair_miner training_data/DenseSolvables training_data/DenseContrastivePairs 100000

echo "=========================================================="
echo " DENSE PIPELINE COMPLETE"
echo "=========================================================="
