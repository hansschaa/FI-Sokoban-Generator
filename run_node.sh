#!/bin/bash
# Uso: ./run_node.sh <NODE_ID> <N_RUNS_PER_NODE>
# Ejemplo para 10 PCs y 30 runs totales (3 runs por PC): ./run_node.sh 1 3

NODE_ID=$1
RUNS_PER_NODE=$2
TIME_LIMIT=5 # Minutos por corrida

if [ -z "$NODE_ID" ] || [ -z "$RUNS_PER_NODE" ]; then
    echo "Uso: ./run_node.sh <NODE_ID (1-10)> <RUNS_PER_NODE>"
    exit 1
fi

echo "Iniciando experimentos en NODO $NODE_ID..."
mkdir -p results_node_$NODE_ID

# Calcular desde qué semilla empieza este nodo
# Ej: Si NODE_ID=1 y RUNS_PER_NODE=3, empieza en la semilla 1 y termina en la 3.
# Si NODE_ID=2, empieza en la semilla 4 y termina en la 6.
START_SEED=$(( (NODE_ID - 1) * RUNS_PER_NODE + 1 ))
END_SEED=$(( START_SEED + RUNS_PER_NODE - 1 ))

ALGORITHMS=("GA" "ES" "SA")
VARIANTS=("--no-surrogate" "") # Vacio significa CON surrogate
SHELLS=(levels/shells/BT_*.txt)

for shell_file in "${SHELLS[@]}"; do
    shell_name=$(basename "$shell_file" .txt)
    
    for alg in "${ALGORITHMS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            
            variant_name="surrogate"
            if [ "$variant" == "--no-surrogate" ]; then
                variant_name="astar"
            fi
            
            for (( seed=$START_SEED; seed<=$END_SEED; seed++ )); do
                output_file="results_node_${NODE_ID}/${shell_name}_${alg}_${variant_name}_seed${seed}.txt"
                echo "Ejecutando: $alg | $shell_name | $variant_name | Seed: $seed"
                
                # Ejecutar y guardar el log
                ./build/evolution_generator $alg pushes 1 --shell $shell_file --time-limit-mins $TIME_LIMIT --seed $seed --no-parallel --show-stats $variant > "$output_file" 2>&1
                
            done
        done
    done
done

echo "¡NODO $NODE_ID FINALIZADO!"
