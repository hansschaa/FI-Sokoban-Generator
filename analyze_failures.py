import csv
from collections import defaultdict

def main():
    csv_file = "benchmark_results_0_to_40.csv"
    meta_file = "benchmark_stratified_heldout_meta.csv"
    
    # Cargar metadatos
    meta = {}
    try:
        with open(meta_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta[int(row['index'])] = row
    except FileNotFoundError:
        print("No se encontró el archivo de metadatos.")

    # Cargar resultados
    data = defaultdict(dict)
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            board_id = int(row['board_id'])
            heur = row['heuristic']
            data[board_id][heur] = row

    failed_boards = []
    
    print(f"\n{'Board':<7} | {'Meta (Pushes/Cajas)':<20} | {'Hungarian (Nodos -> ms)':<25} | {'Massive Fallido (Nodos)':<25} | {'Tipo de Fallo'}")
    print("-" * 115)
    
    for board_id, runs in sorted(data.items()):
        if 'hungarian' not in runs or 'neural_batched_massive' not in runs:
            continue
            
        h_status = runs['hungarian']['status']
        n_status = runs['neural_batched_massive']['status']
        
        if h_status == 'SOLVED' and n_status != 'SOLVED':
            h_nodes = int(runs['hungarian']['expanded_nodes'])
            h_time = float(runs['hungarian']['runtime_ms'])
            
            n_nodes = int(runs['neural_batched_massive']['expanded_nodes'])
            n_time = float(runs['neural_batched_massive']['runtime_ms'])
            
            pushes = meta.get(board_id, {}).get('pushes', '?')
            boxes = meta.get(board_id, {}).get('boxes', '?')
            meta_str = f"P: {pushes} | C: {boxes}"
            
            if n_nodes < h_nodes * 0.5:
                tipo = "Lentitud Computacional (Timeout)"
            else:
                tipo = "Explosion de Nodos / Mala Guia"
            
            print(f"{board_id:<7} | {meta_str:<20} | {h_nodes:>6} n -> {h_time:>6.0f} ms | {n_nodes:>6} n -> {n_time:>6.0f} ms | {tipo}")
            failed_boards.append(board_id)
            
    print(f"\nTotal de tableros analizados donde Hungarian gano y Neural Masivo fallo: {len(failed_boards)}")

if __name__ == "__main__":
    main()
