#!/bin/bash
set -e

SOK_FILE="sok_files/benchmark_stratified_heldout.sok"
END_BOARDS=40

for k in 8 16 32 64; do
    echo "=========================================================="
    echo "  Evaluando BATCH_K=$k"
    echo "=========================================================="
    
    export BATCH_K=$k
    CSV_OUT="benchmark_batchk_${k}.csv"
    
    python3 run_benchmark.py --file $SOK_FILE --end $END_BOARDS || true
    
    if [ -f "benchmark_results_0_to_${END_BOARDS}.csv" ]; then
        mv "benchmark_results_0_to_${END_BOARDS}.csv" "$CSV_OUT"
        echo "--> Guardado en $CSV_OUT"
    else
        echo "Error: no se generó el CSV."
    fi
    echo ""
done

echo "¡Recalibración completada!"
echo "Puedes procesar las intersecciones de cada CSV para comparar nodos y tiempos."
