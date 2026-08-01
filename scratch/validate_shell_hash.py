import torch
import glob
import os
import hashlib

def main():
    print("=== 1. VERIFICANDO KEYS EN DATASET ORIGINAL ===")
    fold1_data = torch.load("results/regressor_fold1_train.pt", map_location='cpu', weights_only=False)
    print(f"Keys presentes en el primer elemento de Fold 1: {list(fold1_data[0].keys())}")
    
    # Extraemos algunos shell_hashes reales del fold 1
    target_hashes = set()
    for item in fold1_data:
        if 'shell_hash' in item:
            target_hashes.add(item['shell_hash'])
            if len(target_hashes) >= 10:
                break
    
    print(f"\nExtraídos 10 shell_hashes de muestra del Fold 1 para búsqueda cruzada.")
    print("=== 2. BÚSQUEDA CRUZADA EN RAW .SOK ===")
    
    directory = "../training_data/Solvables"
    files = glob.glob(os.path.join(directory, "**/*.sok"), recursive=True)
    
    matches = 0
    MOBILE_CHARS = str.maketrans("$.*@+", "     ")
    
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3: continue
            
            board_str = "\n".join(lines[1:])
            
            # Usando la función de hash ORIGINAL de prepare_regressor.py
            shell_str = board_str.translate(MOBILE_CHARS)
            calculated_hash = hashlib.sha256(shell_str.encode()).hexdigest()
            
            if calculated_hash in target_hashes:
                print(f"\n[MATCH ENCONTRADO]")
                print(f"Archivo Origen: {fpath}")
                print(f"Hash Calculado: {calculated_hash}")
                print(f"Pertenece a Fold: 1 (confirmado por regressor_fold1_train.pt)")
                print("--- Tablero ---")
                print(board_str)
                matches += 1
                target_hashes.remove(calculated_hash)
                
            if matches >= 3:
                return

if __name__ == "__main__":
    os.chdir("surrogate_models")
    main()
