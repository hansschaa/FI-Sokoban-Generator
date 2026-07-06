import os

def extract_boards():
    with open('levels/experiments_shells_10.txt', 'r') as f:
        content = f.read()

    blocks = content.split('================================================================================')
    
    os.makedirs('levels/shells', exist_ok=True)
    
    board_idx = 1
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Remove "Shell ID: XXX" line and empty lines
        lines = block.split('\n')
        board_lines = []
        for line in lines:
            if line.startswith('Shell ID:'):
                continue
            if line.strip() == '':
                continue
            board_lines.append(line)
        
        # Write to BT_X.txt
        with open(f'levels/shells/BT_{board_idx}.txt', 'w') as out_f:
            out_f.write('\n'.join(board_lines) + '\n')
            
        board_idx += 1

if __name__ == '__main__':
    extract_boards()
