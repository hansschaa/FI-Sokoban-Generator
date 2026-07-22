#!/bin/bash
RUNS=${1:-500000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/build/es_dataset_generator"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: No se encontro el binario en $BINARY"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/sokoban_dataset_buckets"
cd "$SCRIPT_DIR"

if pgrep -f "es_dataset_generator" > /dev/null; then
    echo "AVISO: El miner ya esta corriendo."
    exit 1
fi

echo "Iniciando Sokoban Miner... Runs: $RUNS"

# stdbuf -oL fuerza salida linea por linea (visible en tiempo real en el log)
nohup stdbuf -oL "$BINARY" --runs $RUNS > miner_log.txt 2>&1 &
MINER_PID=$!

echo "PID: $MINER_PID"
echo "Log en vivo:  tail -f $SCRIPT_DIR/miner_log.txt"
echo "Progreso:     grep -c 'pushes:' $SCRIPT_DIR/sokoban_dataset_buckets/*.sok 2>/dev/null"
echo "Detener:      kill $MINER_PID"
echo $MINER_PID > "$SCRIPT_DIR/miner.pid"
