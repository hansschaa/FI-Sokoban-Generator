#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$DIR/.."

echo "Extrayendo un tablero de prueba..."
awk '/^#/{print; flag=1; next} /^$/{if(flag) exit} flag' $BASE_DIR/sok_files/auto_generated.sok > $BASE_DIR/tests/test_board.txt

echo "Ejecutando Python (surrogate_models/data/board_utils.py)..."
$BASE_DIR/venv/bin/python $BASE_DIR/tests/test_tensor_parity.py > $BASE_DIR/tests/py_out.txt

echo "Ejecutando C++ (src/neural_heuristic.cpp)..."
$BASE_DIR/build/test_tensor_parity $BASE_DIR/tests/test_board.txt > $BASE_DIR/tests/cpp_out.txt

echo "Comparando..."
diff $BASE_DIR/tests/py_out.txt $BASE_DIR/tests/cpp_out.txt

echo "✅ TEST PASSED: La paridad de tensores (Los 6 canales, incluyendo deadlocks y posicionamiento espacial) entre C++ y Python es perfecta."

# Limpieza
rm $BASE_DIR/tests/test_board.txt $BASE_DIR/tests/py_out.txt $BASE_DIR/tests/cpp_out.txt
