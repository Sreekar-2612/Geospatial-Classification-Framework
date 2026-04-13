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
        if 'cam = GradCAM' not in cell.source:
            cell.source = cell.source.replace(
                'error_indices = np.where(y_true != y_pred)[0]',
                'error_indices = np.where(y_true != y_pred)[0]\n\n# Ensure GradCAM is actively bound to the layer\ntarget_layer = cnn_model.layer4[-1].conv2\ncam = GradCAM(cnn_model, target_layer)'
            )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Injected the standalone `cam` definition directly into the error analysis cell!")
