import nbformat as nbf

notebook_path = 'notebooks/05_advanced_analysis.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
except Exception as e:
    print(f"Error opening notebook: {e}")
    exit(1)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Replace the hardcoded CLASSES list with the correct 5 classes
        if 'CLASSES = ["Agriculture", "Buildings", "Desert"' in cell.source:
            cell.source = cell.source.replace(
                'CLASSES = ["Agriculture", "Buildings", "Desert", "Forest", "Roads", "Water"]',
                'CLASSES = ["Agriculture", "Buildings", "Forest", "Roads", "Water"]'
            )
            # Replace the Color map dictionary specifically removing Desert
            cell.source = cell.source.replace(
                '"Agriculture": "#639922", "Buildings": "#888780", "Desert": "#BA7517",\n    "Forest": "#1D9E75"',
                '"Agriculture": "#639922", "Buildings": "#888780",\n    "Forest": "#1D9E75"'
            )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Successfully fixed! 'Desert' class removed from your notebook.")
