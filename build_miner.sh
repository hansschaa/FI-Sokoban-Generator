#!/bin/bash
# build_miner.sh - Compila nativamente en la maquina destino

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Instalando compilador y herramientas de construccion..."
sudo apt update && sudo apt install -y build-essential cmake gcc g++

echo "Creando directorio de compilacion..."
mkdir -p build
cd build

echo "Configurando con CMake..."
cmake ..

echo "Compilando..."
make es_dataset_generator -j$(nproc)

if [ -f "es_dataset_generator" ]; then
    echo "========================================="
    echo "✅ COMPILACION COMPLETADA CON EXITO."
    echo "========================================="
else
    echo "❌ ERROR EN LA COMPILACION."
    exit 1
fi
