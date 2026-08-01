with open("surrogate_models/prepare_path_consistency.py", "r") as f:
    content = f.read()
    
# Find simulate_path
import re
new_content = content.replace("        target_char = lines[bx][by]", """        if bx < 0 or by < 0 or bx >= len(lines) or by >= len(lines[bx]):
            return []
        target_char = lines[bx][by]""")

with open("surrogate_models/prepare_path_consistency.py", "w") as f:
    f.write(new_content)
