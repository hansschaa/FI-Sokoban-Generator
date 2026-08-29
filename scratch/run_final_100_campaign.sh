#!/bin/bash
echo "==========================================================="
echo " INICIANDO CAMPAÑA DE 100 CORRIDAS (FULL SURROGATE Y HYBRID)"
echo "==========================================================="

echo "Matando servidores Flask previos..."
pkill -f surrogate_server.py
sleep 2

echo "Levantando Flask en background..."
venv/bin/python3 surrogate_models/surrogate_server.py > scratch/flask_campaign.log 2>&1 &
FLASK_PID=$!
sleep 5

echo "Iniciando script Python (run_exp1_neural_update.py)..."
venv/bin/python3 scratch/run_exp1_neural_update.py

echo "Limpiando servidor Flask..."
kill $FLASK_PID
pkill -f surrogate_server.py
echo "==========================================================="
echo " CAMPAÑA COMPLETADA"
echo "==========================================================="
