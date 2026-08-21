import os
import glob
import re

def main():
    print("=" * 60)
    print("  FUSIONANDO Y ETIQUETANDO DATASET DE WORKERS")
    print("=" * 60)

    out_dir = "training_data/DenseSolvables"
    os.makedirs(out_dir, exist_ok=True)

    # Limpiar el directorio destino de archivos PC anteriores por las dudas
    for old_file in glob.glob(os.path.join(out_dir, "*_pc*.sok")):
        os.remove(old_file)

    for pc_id in [1, 2, 3]:
        worker_dir = f"training_data/DenseSolvables_PC{pc_id}"
        if not os.path.exists(worker_dir):
            print(f"⚠️ No se encontró {worker_dir}, omitiendo.")
            continue
            
        sok_files = glob.glob(os.path.join(worker_dir, "*.sok"))
        total_boards = 0
        
        for fpath in sok_files:
            filename = os.path.basename(fpath)
            # Renombrar para no pisarse (ej. 11_to_20.sok -> 11_to_20_pc1.sok)
            out_filename = filename.replace(".sok", f"_pc{pc_id}.sok")
            out_path = os.path.join(out_dir, out_filename)
            
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
            
            with open(out_path, "w", encoding="utf-8") as f_out:
                for block in blocks:
                    lines = block.splitlines()
                    if len(lines) < 2: continue
                    
                    header = lines[0]
                    board_lines = lines[1:]
                    
                    # Inyectar la etiqueta de trazabilidad en el header
                    new_header = header + f" source_pc:PC{pc_id}"
                    
                    f_out.write(new_header + "\n")
                    f_out.write("\n".join(board_lines) + "\n\n")
                    total_boards += 1
                    
        print(f"✅ PC{pc_id}: {total_boards} tableros etiquetados y movidos a {out_dir}")

    print("\n¡Fusión completada con trazabilidad intacta!")

if __name__ == "__main__":
    main()
