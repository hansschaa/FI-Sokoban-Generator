import os
import glob
import re

def fix_sok_files():
    directory = 'training_data/Solvables'
    print(f"Buscando archivos .sok en {directory}...")
    sok_files = glob.glob(os.path.join(directory, '**', '*.sok'), recursive=True)
    
    # Expresión regular para buscar 'pushes:X' en cualquier lugar
    pattern = re.compile(r'(pushes:)(\d+)')
    
    def repl(match):
        val = int(match.group(2))
        new_val = max(0, val - 1)
        return f"pushes:{new_val}"
    
    processed = 0
    modified = 0
    
    for filepath in sok_files:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        new_content = pattern.sub(repl, content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            modified += 1
        processed += 1
        
    print(f"Procesados {processed} archivos.")
    print(f"Modificados {modified} archivos para restar 1 a los pushes.")

if __name__ == "__main__":
    fix_sok_files()
