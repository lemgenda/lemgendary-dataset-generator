import os
import json
import re
from pathlib import Path

def resolve_model_name(folder_name):
    # LemGendizedNimaAestheticLarge -> nima_aesthetic
    clean_name = folder_name.replace("LemGendized", "").replace("KaggleReady", "").replace("Large", "").replace("Mini", "")
    # PascalCase to snake_case
    resolved = re.sub(r'(?<!^)(?=[A-Z])', '_', clean_name).lower()
    
    # Standardize common keys
    if "naf_net" in resolved: resolved = resolved.replace("naf_net", "nafnet")
    if "upn_v_2" in resolved: resolved = resolved.replace("upn_v_2", "upn_v2")
    return resolved

def fix_notebook(path):
    print(f"Fixing {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        folder_name = path.parent.name
        model_name = resolve_model_name(folder_name)
        print(f"   -> Resolved Model: {model_name}")
        
        changed = False
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                new_source = []
                for line in source:
                    if "[MODEL_NAME]" in line:
                        line = line.replace("[MODEL_NAME]", model_name)
                        changed = True
                    # Also fix the descriptive note if it exists
                    if "NOTE: Replace [MODEL_NAME]" in line:
                        line = line.replace("Replace [MODEL_NAME]", f"Replace '{model_name}'")
                        changed = True
                    new_source.append(line)
                cell['source'] = new_source
        
        if changed:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            print(f"   Updated!")
        else:
            print(f"   No changes needed.")
            
    except Exception as e:
        print(f"   Error: {e}")

def main():
    root = Path(r"C:\Development\python\model-training\LemGendaryDatasets")
    print(f"Scanning {root} for notebooks (Fast Scan)...")
    
    # Efficient Scan: only check root and 1st level subdirs
    notebooks = list(root.glob("*.ipynb"))
    for subdir in root.iterdir():
        if subdir.is_dir():
            notebooks.extend(list(subdir.glob("*.ipynb")))
            
    print(f"Found {len(notebooks)} notebooks.")
    
    for nb in notebooks:
        fix_notebook(nb)

if __name__ == "__main__":
    main()
