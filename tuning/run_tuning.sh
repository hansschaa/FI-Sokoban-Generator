#!/bin/bash
# =============================================================================
# run_tuning.sh  —  Lanzador unificado de sintonización con irace
# Uso: ./run_tuning.sh GA|ES|SA
# Ejecutar desde el directorio tuning/
# =============================================================================

set -e

# ADVERTENCIA DE CONCURRENCIA:
# NO ejecutar dos instancias de este script en el mismo directorio al mismo tiempo.
# Ambas instancias modifican 'target-runner' con sed (ALGO= y FO=).
# Cada computadora debe correr UN solo algoritmo (GA, ES o SA).

ALGO="${1^^}"   # Normalizar a mayúsculas

if [[ "$ALGO" != "GA" && "$ALGO" != "ES" && "$ALGO" != "SA" ]]; then
    echo "[ERROR] Uso: $0 GA|ES|SA"
    exit 1
fi

echo "======================================================================"
echo " SINTONIZACIÓN IRACE — Algoritmo: $ALGO"
echo " Criterio de término: maxEvals=1000 | stagLimit=200 (igual para todos)"
echo " FOs a tunear: FO1, FO4, FO5"
echo "======================================================================"

cd "$(dirname "$0")"

# Seleccionar archivo de parámetros según el algoritmo
case "$ALGO" in
    GA) PARAM_FILE="GA_parameters.txt" ;;
    ES) PARAM_FILE="ES_parameters.txt" ;;
    SA) PARAM_FILE="SA_parameters.txt" ;;
esac

echo "[OK] Archivo de parámetros: $PARAM_FILE"

# Verificar que el binario existe
if [ ! -f "../build/irace_generator" ]; then
    echo "[ERROR] No existe ../build/irace_generator. Compila primero:"
    echo "        cd ../build && make irace_generator -j\$(nproc)"
    exit 1
fi

# Iterar sobre las 3 funciones objetivo
for FO in FO1 FO4 FO5; do
    echo ""
    echo "-------------------------------------------------------------------"
    echo " Sintonizando $ALGO para $FO ..."
    echo "-------------------------------------------------------------------"

    # Configurar scenario.txt
    sed -i "s|^parameterFile.*|parameterFile = \"$PARAM_FILE\"|" scenario.txt

    # Configurar target-runner (ALGO y FO)
    sed -i "s|^ALGO=.*|ALGO=\"$ALGO\"|"   target-runner
    sed -i "s|^FO=.*|FO=\"$FO\"|"         target-runner

    # Limpiar Rdata anterior para evitar contaminación
    rm -f irace.Rdata

    # Lanzar irace
    irace --parallel 24

    # Guardar resultado
    OUTFILE="../tuning_results/irace_${ALGO}_${FO}_fixed.Rdata"
    mv irace.Rdata "$OUTFILE"

    echo "[OK] Resultado guardado en: $OUTFILE"
done

echo ""
echo "======================================================================"
echo " ¡Sintonización de $ALGO completada para FO1, FO4, FO5!"
echo " Para extraer los parámetros óptimos, ejecuta en R:"
echo "   Rscript ../scripts/extract_irace_params.R $ALGO"
echo "======================================================================"
