import papermill as pm
import nbformat
import glob
from pathlib import Path

# buggy notebooks
exclusion_list_basic = [
    "Part 10 - Federated Learning with Secure Aggregation.ipynb",
    "Part 13b - Secure Classification with Syft Keras and TFE - Secure Model Serving.ipynb",
    "Part 13c - Secure Classification with Syft Keras and TFE - Private Prediction Client.ipynb",
]

exclusion_list_advanced = [
    "Build your own tensor type (advanced).ipynb",
    "Federated Recurrent Neural Network.ipynb",
]


def test_notebooks_basic(isolated_filesystem):
    notebooks = glob.glob("*.ipynb")
    for notebook in notebooks:
        if Path(notebook).name in exclusion_list_basic:
            continue
        print(notebook)
        res = pm.execute_notebook(
            notebook, "/dev/null", parameters={"epochs": 1, "n_test_batches": 5}, timeout=300
        )
        assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_advanced(isolated_filesystem):
    notebooks = glob.glob("advanced/*.ipynb")
    for notebook in notebooks:
        if Path(notebook).name in exclusion_list_advanced:
            continue
        print(notebook)
        res = pm.execute_notebook(notebook, "/dev/null", parameters={"epochs": 1}, timeout=300)
        assert isinstance(res, nbformat.notebooknode.NotebookNode)
