#!/bin/bash
echo "Iniciando sintonización (Vía Larga) para Simulated Annealing (SA)"
echo "------------------------------------------------------------------"

cd "$(dirname "$0")"

# Archivo de parámetros
PARAM_FILE="SA_parameters.txt"

# Asegurarse de que scenario.txt use el parámetro de SA
sed -i 's/parameterFile.*/parameterFile = "'"$PARAM_FILE"'"/' scenario.txt

for FO in FO1 FO4 FO5; do
    echo "====================================================="
    echo "Sintonizando SA para $FO ..."
    echo "====================================================="
    
    # Actualizar target-runner
    sed -i 's/^ALGO=.*/ALGO="SA"/' target-runner
    sed -i 's/^FO=.*/FO="'"$FO"'"/' target-runner
    
    # Correr irace
    irace --parallel 24
    
    # Guardar resultado
    mv irace.Rdata "../tuning_results/irace_SA_${FO}_fixed.Rdata"
    
    echo "Sintonización para $FO completada. Resultado guardado en tuning_results/irace_SA_${FO}_fixed.Rdata"
    echo ""
done

echo "¡Todas las sintonizaciones de SA finalizaron!"
echo "Abre R, extrae los parámetros óptimos, e ingrésalos en scripts/run_experiment_rq1_rq2.py"
