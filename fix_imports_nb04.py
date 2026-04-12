import nbformat as nbf

notebook_path = 'notebooks/04_deep_learning.ipynb'
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit(1)

for cell in nb.cells:
    if cell.cell_type == 'code' and 'Isolating Highest Confidence Misclassifications...' in cell.source:
        if 'import numpy as np' not in cell.source:
            cell.source = cell.source.replace(
                'import torch.nn.functional as F',
                'import torch.nn.functional as F\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport cv2'
            )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Injected required imports directly into the notebook cell!")
