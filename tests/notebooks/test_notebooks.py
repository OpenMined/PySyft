import os
from pathlib import Path

import nbformat
import papermill as pm

from .. import GRID_NETWORK_PORT, worker_ports

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
examples_path = dir_path.parent.parent

data_centric_mnist_path = examples_path.joinpath("examples", "data-centric", "mnist")
data_centric_intro_path = examples_path.joinpath(
    "examples", "data-centric", "introduction"
)

model_centric_mnist_path = examples_path.joinpath("examples", "model-centric")


def test_notebooks_data_centric_mnist_01():
    """Test if notebook r"""
    notebook_mnist_01 = data_centric_mnist_path.joinpath(
        "01-FL-mnist-populate-a-grid-node.ipynb"
    )
    res = pm.execute_notebook(
        str(notebook_mnist_01),
        os.devnull,
        dict(
            alice_address=("http://localhost:" + worker_ports["Alice"]),
            bob_address=("http://localhost:" + worker_ports["Bob"]),
        ),
    )

    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_data_centric_mnist_02():
    notebook_mnist_02 = data_centric_mnist_path.joinpath(
        "02-FL-mnist-train-model.ipynb"
    )

    res = pm.execute_notebook(
        str(notebook_mnist_02),
        "kk.ipynb",
        dict(
            grid_address="http://localhost:" + GRID_NETWORK_PORT, N_EPOCHS=2, N_TEST=2
        ),
    )

    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_data_centric_introduction():
    notebook_intro_00 = data_centric_intro_path.joinpath(
        "01-introduction-to-pygrid.ipynb"
    )

    res = pm.execute_notebook(
        str(notebook_intro_00),
        os.devnull,
        dict(
            grid_address=("http://localhost:" + GRID_NETWORK_PORT),
            bob=("http://localhost:" + worker_ports["Bob"]),
        ),
    )

    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_model_centric_mnist_01():
    nb_model_centric_mnist_01 = model_centric_mnist_path.joinpath(
        "01-Create-plan.ipynb"
    )

    res = pm.execute_notebook(
        str(nb_model_centric_mnist_01),
        os.devnull,
        dict(gridAddress=("localhost:" + worker_ports["Alice"])),
    )

    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_model_centric_mnist_02():
    nb_model_centric_mnist_02 = model_centric_mnist_path.joinpath(
        "02-ExecutePlan.ipynb"
    )

    res = pm.execute_notebook(
        str(nb_model_centric_mnist_02),
        os.devnull,
        dict(gridAddress=("ws://localhost:" + worker_ports["Alice"])),
    )

    assert isinstance(res, nbformat.notebooknode.NotebookNode)
