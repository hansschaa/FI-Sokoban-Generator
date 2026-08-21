import os
import glob

def main():
    print("=" * 60)
    print("  FUSIONANDO Y ETIQUETANDO DATASET DE WORKERS")
    print("=" * 60)

    out_dir = "training_data/DenseSolvables"
    os.makedirs(out_dir, exist_ok=True)

    # Limpiar el directorio destino de archivos PC/GTX anteriores por las dudas
    for old_file in glob.glob(os.path.join(out_dir, "*_src_*.sok")):
        os.remove(old_file)
    for old_file in glob.glob(os.path.join(out_dir, "*_pc*.sok")):
        os.remove(old_file)

    # Buscar todas las carpetas que empiecen con DenseSolvables_
    base_data_dir = "training_data"
    worker_dirs = [d for d in os.listdir(base_data_dir) 
                   if os.path.isdir(os.path.join(base_data_dir, d)) 
                   and d.startswith("DenseSolvables_")]

    if not worker_dirs:
        print("⚠️ No se encontraron carpetas de workers (ej. DenseSolvables_GTX3, DenseSolvables_PC1).")
        return

    for w_dir in worker_dirs:
        # Extraer el identificador de la maquina (ej. GTX3 de DenseSolvables_GTX3)
        machine_id = w_dir.replace("DenseSolvables_", "")
        worker_path = os.path.join(base_data_dir, w_dir)
            
        sok_files = glob.glob(os.path.join(worker_path, "*.sok"))
        total_boards = 0
        
        for fpath in sok_files:
            filename = os.path.basename(fpath)
            # Renombrar para no pisarse (ej. 11_to_20.sok -> 11_to_20_src_GTX3.sok)
            out_filename = filename.replace(".sok", f"_src_{machine_id}.sok")
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
                    new_header = header + f" source_pc:{machine_id}"
                    
                    f_out.write(new_header + "\n")
                    f_out.write("\n".join(board_lines) + "\n\n")
                    total_boards += 1
                    
        print(f"✅ {machine_id}: {total_boards} tableros etiquetados y movidos a {out_dir}")

    print("\n¡Fusión completada con trazabilidad intacta!")

if __name__ == "__main__":
    main()
