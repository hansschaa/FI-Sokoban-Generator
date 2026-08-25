import subprocess
import os
import tempfile
import pandas as pd

def extract_boards(sok_file):
    with open(sok_file, 'r') as f: content = f.read()
    boards = []
    current_board = []
    board_id = 0
    for line in content.split('\n'):
        if line.strip().startswith('#'):
            current_board.append(line)
        elif line.strip() == '' and current_board:
            boards.append((board_id, '\n'.join(current_board)))
            current_board = []
            board_id += 1
    if current_board:
        boards.append((board_id, '\n'.join(current_board)))
    return boards

def main():
    target_ids = [18, 35]
    pf = 'sok_files/benchmark_stratified_heldout.sok'
    boards = extract_boards(pf)
    hard_boards = [b for b in boards if b[0] in target_ids]
    
    fd, temp_sok = tempfile.mkstemp(prefix="hard_", suffix=".sok")
    with os.fdopen(fd, 'w') as f:
        for bid, bstr in hard_boards:
            f.write(bstr + "\n\n")
            
    print(f"Tableros 18 y 35 guardados en {temp_sok}")
    
    heuristics = ["manhattan", "hungarian", "neural_sequential", "neural_batched_massive"]
    out_csv = "hard_boards_results.csv"
    with open(out_csv, "w") as f:
        f.write("board_id,heuristic,status,runtime_ms,pushes,expanded_nodes\n")
        
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    
    for h in heuristics:
        print(f"--- Evaluando {h} ---")
        fd_out, temp_out = tempfile.mkstemp(prefix="out_", suffix=".tsv")
        os.close(fd_out)
        
        cmd = ["./build/batch_solver", temp_sok, h, temp_out]
        subprocess.run(cmd, env=env, check=True)
        
        with open(temp_out, "r") as f:
            lines = f.read().strip().split('\n')[1:] # skip header
            
        with open(out_csv, "a") as f:
            for idx, line in enumerate(lines):
                if not line.strip(): continue
                parts = line.split('\t')
                board_id = hard_boards[idx][0]
                status = parts[1]
                runtime = parts[3]
                pushes = parts[4]
                nodes = parts[7]
                f.write(f"{board_id},{h},{status},{runtime},{pushes},{nodes}\n")
                
        os.remove(temp_out)
        
    os.remove(temp_sok)
    
    df = pd.read_csv(out_csv)
    print("\nResultados Finales:")
    print(df.to_string())

if __name__ == '__main__':
    main()
