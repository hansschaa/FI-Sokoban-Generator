import os
import torch
import glob

OUTPUT_DIR = "results/path_consistency"
TOTAL_PARTS = 4

def merge_parts():
    print(f"Buscando archivos en {OUTPUT_DIR}...")
    for k in range(1, 6):
        combined_dataset = []
        part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"path_fold{k}_train_part*.pt")))
        
        if not part_files:
            print(f"ADVERTENCIA: No se encontraron partes para el Fold {k}")
            continue
            
        for part_file in part_files:
            print(f"Cargando {part_file}...")
            data = torch.load(part_file, map_location='cpu', weights_only=False)
            combined_dataset.extend(data)
        
        if len(combined_dataset) > 0:
            out_file = os.path.join(OUTPUT_DIR, f"path_fold{k}_train.pt")
            print(f"==> Guardando {out_file} con {len(combined_dataset)} pares en total.")
            torch.save(combined_dataset, out_file)
        else:
            print(f"Fold {k} vacío. No se guardó nada.")

if __name__ == "__main__":
    merge_parts()
    print("Fusión completada. ¡Puedes borrar los archivos _partX.pt para liberar espacio!")
