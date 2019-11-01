import papermill as pm
import nbformat
import pytest
from pathlib import Path
from torchvision import datasets
import os

notebooks_to_run = (
    ["examples/tutorials/Part 01 - The Basic Tools of Private Deep Learning.ipynb", {}],
    ["examples/tutorials/Part 02 - Intro to Federated Learning.ipynb", {}],
    ["examples/tutorials/Part 03 - Advanced Remote Execution Tools.ipynb", {}],
    ["examples/tutorials/Part 04 - Federated Learning via Trusted Aggregator.ipynb", {}],
    ["examples/tutorials/Part 05 - Welcome to the Sandbox.ipynb", {}],
    ["examples/tutorials/Part 06 - Federated Learning on MNIST using a CNN.ipynb", {"epochs": 1}],
    ["examples/tutorials/Part 07 - Federated Learning with Federated Dataset.ipynb", {}],
    ["examples/tutorials/Part 08 - Introduction to Plans.ipynb", {}],
    ["examples/tutorials/Part 08 bis - Introduction to Protocols.ipynb", {}],
    ["examples/tutorials/Part 09 - Intro to Encrypted Programs.ipynb", {}],
    [
        "examples/tutorials/Part 11 - Secure Deep Learning Classification.ipynb",
        {"epochs": 1, "n_test_batches": 5},
    ],
    ["examples/tutorials/Part 12 - Train an Encrypted Neural Network on Encrypted Data.ipynb", {}],
    ["examples/tutorials/Part 12 bis - Encrypted Training on MNIST.ipynb", {"epochs": 1}],
)


@pytest.mark.parametrize("notebook,parameters", notebooks_to_run)
def test_notebooks(notebook, parameters, isolated_filesystem):
    print(notebook)
    res = pm.execute_notebook(notebook, "/dev/null", parameters=parameters)
    print(type(res))
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
