#!/bin/bash
set -e

echo "=========================================================="
echo " PREPARING AND BALANCING DATASET"
echo "=========================================================="
# Requires virtual environment
source venv/bin/activate || true

cd surrogate_models/data
python prepare_contrastive_classifier.py
cd ../..

echo "=========================================================="
echo " RETRAINING CONTRASTIVE CLASSIFIER (5-FOLD CV)"
echo "=========================================================="
cd surrogate_models
python train_contrastive_classifier_cv.py
cd ..

echo "=========================================================="
echo " RETRAINING COMPLETE"
echo "=========================================================="
