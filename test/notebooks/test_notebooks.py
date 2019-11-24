import papermill as pm
import nbformat
import glob
from pathlib import Path

# buggy notebooks
exclusion_list = [
    "Part 10 - Federated Learning with Secure Aggregation.ipynb",
    "Part 13b - Secure Classification with Syft Keras and TFE - Secure Model Serving.ipynb",
    "Part 13c - Secure Classification with Syft Keras and TFE - Private Prediction Client.ipynb",
]

# Data Dependent notebooks
# Part 06 - Federated Learning on MNIST using a CNN (MNIST in '../data')
# Part 10 ('../data/BostonHousing/boston_housing.pickle')
# Part 11 (MNIST in '../data')
# Part 12 bis - Encrypted Training on MNIST (MNIST in '../data')


def test_notebooks(isolated_filesystem):
    notebooks = glob.glob("*.ipynb")
    for notebook in notebooks:
        if Path(notebook).name in exclusion_list:
            continue
        print(notebook)
        res = pm.execute_notebook(
            notebook, "/dev/null", parameters={"epochs": 1, "n_test_batches": 5}, timeout=300
        )
        assert isinstance(res, nbformat.notebooknode.NotebookNode)
