import os
import subprocess
import sys
from datetime import datetime

def main():
    print("========================================")
    print(" INICIANDO EXPERIMENTO 1 (HORNO DE SIMULACIÓN)")
    print("========================================")

    csv_filename = "exp1_raw_data.csv"
    
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w") as f:
            f.write("FO,BT_id,Algoritmo,Semilla_ID,Fitness_Bruto,Board_Hash,Board_String,Tiempo_Segundos\n")
        print(f"[*] Archivo {csv_filename} creado con las cabeceras.")
    else:
        print(f"[*] Archivo {csv_filename} ya existe. Los resultados se anexarán.")

    objetivos = ["FO1", "FO4", "FO5"]
    fo_map = {"FO1": "Pushes", "FO4": "Deadlocks", "FO5": "NodosRepetidos"}
    tableros = list(range(1, 11)) # BT_01 a BT_10
    algoritmos = ["SA", "ES", "GA"]
    repeticiones = list(range(1, 16)) # 1 a 15

    total_runs = len(objetivos) * len(tableros) * len(algoritmos) * len(repeticiones)
    current_run = 0

    binary_path = "./build/experiment_runner"
    
    if not os.path.exists(binary_path):
        print(f"[!] ERROR: El ejecutable {binary_path} no existe. Por favor compila primero.")
        sys.exit(1)

    start_time = datetime.now()

    for fo in objetivos:
        for bt in tableros:
            for alg in algoritmos:
                for run in repeticiones:
                    current_run += 1
                    
                    board_file = f"levels/shells/BT_{bt}.txt"
                    cmd = [binary_path, alg, fo, str(run), board_file]
                    
                    print(f"[{current_run}/{total_runs}] Ejecutando: {fo_map[fo]} | BT_{bt:02d} | {alg} | Run {run} ...", end="", flush=True)
                    
                    try:
                        start_cmd = datetime.now()
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        end_cmd = datetime.now()
                        duration_sec = (end_cmd - start_cmd).total_seconds()
                        
                        if result.returncode == 0:
                            lines = [line for line in result.stdout.strip().split('\n') if ';' in line]
                            last_line = lines[-1] if lines else ""
                            output = last_line.strip().split(";")
                            if len(output) >= 3:
                                fitness = output[0].strip()
                                b_hash = output[1].strip()
                                b_str = output[2].strip()
                                with open(csv_filename, "a") as f:
                                    f.write(f"{fo_map[fo]},{bt},{alg},{run},{fitness},{b_hash},{b_str},{duration_sec:.4f}\n")
                                print(f" OK ({duration_sec:.2f}s)")
                            else:
                                print(f" ERROR (formato inesperado: {last_line})")
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
