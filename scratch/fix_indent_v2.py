
import os

path = 'compiler-pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    ln = i + 1
    # Indent lines 1057 to 1410 by 4 more spaces to bring them inside the 'with executor' block
    if 1057 <= ln <= 1410:
        new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("SUCCESS")
