
import os

path = 'compiler-pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    ln = i + 1
    # Shift lines 956 to 1024 right by 4 spaces
    if 956 <= ln <= 1024:
        new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("SUCCESS")
