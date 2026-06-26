import os
import subprocess
import sys
from datetime import datetime

def main():
    print("========================================")
    print(" INICIANDO EXPERIMENTO 1 (HORNO DE SIMULACIÓN)")
    print("========================================")

    csv_filename = "exp1_raw_data.csv"
    
    # Si el archivo no existe o queremos sobreescribirlo, escribimos la cabecera
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w") as f:
            f.write("FO,BT_id,Algoritmo,Semilla_ID,Fitness_Bruto\n")
        print(f"[*] Archivo {csv_filename} creado con las cabeceras.")
    else:
        print(f"[*] Archivo {csv_filename} ya existe. Los resultados se anexarán.")

    objetivos = ["Pushes", "Deadlocks", "NodosRepetidos"]
    tableros = list(range(1, 21)) # BT_01 a BT_20
    algoritmos = ["SA", "ES", "GA"]
    repeticiones = list(range(1, 31)) # 1 a 30

    total_runs = len(objetivos) * len(tableros) * len(algoritmos) * len(repeticiones)
    current_run = 0

    binary_path = "./build/experiment_1"
    
    if not os.path.exists(binary_path):
        print(f"[!] ERROR: El ejecutable {binary_path} no existe. Por favor compila primero.")
        sys.exit(1)

    start_time = datetime.now()

    # Bucle cuádruple anidado
    for fo in objetivos:
        for bt in tableros:
            for alg in algoritmos:
                for run in repeticiones:
                    current_run += 1
                    
                    cmd = [binary_path, fo, str(bt), alg, str(run)]
                    
                    # Para evitar spam en la consola, no mostramos la salida estandar completa
                    # Solo un indicador de progreso por consola.
                    print(f"[{current_run}/{total_runs}] Ejecutando: {fo} | BT_{bt:02d} | {alg} | Run {run} ...", end="", flush=True)
                    
                    try:
                        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                        if result.returncode == 0:
                            print(" OK")
                        else:
                            print(" ERROR")
                            print(f"[!] Mensaje de error: {result.stderr}")
                    except Exception as e:
                        print(f" EXCEPCIÓN: {e}")

    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n========================================")
    print(" EXPERIMENTO COMPLETADO")
    print(f" Ejecuciones totales: {total_runs}")
    print(f" Tiempo total de ejecución: {duration}")
    print(f" Resultados guardados en: {csv_filename}")
    print("========================================")

if __name__ == "__main__":
    main()
