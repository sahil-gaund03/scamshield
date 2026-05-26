import json
import os

notebooks_dir = "notebooks"
for filename in sorted(os.listdir(notebooks_dir)):
    if filename.endswith(".ipynb"):
        filepath = os.path.join(notebooks_dir, filename)
        print(f"\n==========================================")
        print(f"FILE: {filename}")
        print(f"==========================================")
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for i, cell in enumerate(data.get("cells", [])):
                    if cell.get("cell_type") == "code":
                        source = "".join(cell.get("source", []))
                        if source.strip():
                            print(f"--- Cell {i} ---")
                            print(source)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
