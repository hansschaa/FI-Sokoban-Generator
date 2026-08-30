#!/bin/bash
echo "Ejecutando Run 1..."
./build/experiment_runner ES FO1 42 levels/shell_1.sok --heuristic full_surrogate --timeLimit 60 --maxEvals 1000000 --mu 9 --lambda 28 --mutRate 0.8559 --stagLimit 199 > scratch/repro1.log 2>&1
echo "Ejecutando Run 2..."
./build/experiment_runner ES FO1 42 levels/shell_1.sok --heuristic full_surrogate --timeLimit 60 --maxEvals 1000000 --mu 9 --lambda 28 --mutRate 0.8559 --stagLimit 199 > scratch/repro2.log 2>&1

echo "Diferencias encontradas (Filtrando MS, Elapsed, Tiempo):"
diff <(grep -i -v -E "tiempo|ms|elapsed" scratch/repro1.log) <(grep -i -v -E "tiempo|ms|elapsed" scratch/repro2.log)
