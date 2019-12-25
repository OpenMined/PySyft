import glob
import os
import sys
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import pytest
import nbformat
import numpy as np
import pandas as pd
import papermill as pm
import torch

import syft as sy
from syft import TorchHook
from syft.workers.websocket_server import WebsocketServerWorker

# lets start by finding all notebooks currently available in examples and subfolders
all_notebooks = [Path(n) for n in glob.glob("examples/tutorials/**/*.ipynb", recursive=True)]

# buggy notebooks with explanation what does not work
exclusion_list_notebooks = [
    # Part 10 needs either torch.log2 to be implemented or numpy to be hooked
    "Part 10 - Federated Learning with Secure Aggregation.ipynb",
    # Part 13b and c need fixing of the tensorflow serving with PySyft
    "Part 13b - Secure Classification with Syft Keras and TFE - Secure Model Serving.ipynb",
    "Part 13c - Secure Classification with Syft Keras and TFE - Private Prediction Client.ipynb",
    # This notebook is excluded as it needs library code modification which I might add later on
    "Build your own tensor type (advanced).ipynb",
    "Federated Recurrent Neural Network.ipynb",
]

exclusion_list_folders = [
    "examples/tutorials/websocket",
    "examples/tutorials/advanced/Monitor_Network_Traffic",
    "examples/tutorials/advanced/websockets-example-MNIST-parallel",
    # This notebook is skipped because it fails in travis and we do not know why for the moment
    "examples/tutorials/advanced/Federated SMS Spam prediction",
]

# remove known buggy notebooks and folders that should be excluded
not_excluded_notebooks = []
for n in all_notebooks:
    if n.name in exclusion_list_notebooks:
        continue
    elif str(n.parent) in exclusion_list_folders:
        continue
    else:
        not_excluded_notebooks.append(n)


def test_notebooks_basic(isolated_filesystem):
    """Test Notebooks in the tutorial root folder."""
    notebooks = glob.glob("*.ipynb")
    for notebook in notebooks:
        list_name = Path("examples/tutorials/") / notebook
        if list_name in not_excluded_notebooks:
            not_excluded_notebooks.remove(list_name)
            res = pm.execute_notebook(
                notebook,
                "/dev/null",
                parameters={
                    "epochs": 1,
                    "n_test_batches": 5,
                    "n_train_items": 64,
                    "n_test_items": 64,
                },
                timeout=300,
            )
            assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_basic_translations(isolated_filesystem):
    """Test Notebooks in the tutorial root folder."""
    notebooks = glob.glob("translations/**/*.ipynb", recursive=True)
    for notebook in notebooks:
        list_name = Path("examples/tutorials/") / notebook
        if list_name in not_excluded_notebooks:
            not_excluded_notebooks.remove(list_name)
            res = pm.execute_notebook(
                notebook,
                "/dev/null",
                parameters={
                    "epochs": 1,
                    "n_test_batches": 5,
                    "n_train_items": 64,
                    "n_test_items": 64,
                },
                timeout=300,
            )
            assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_advanced(isolated_filesystem):
    notebooks = glob.glob("advanced/*.ipynb")
    notebooks += glob.glob("advanced/Split Neural Network/*.ipynb")
    for notebook in notebooks:
        list_name = Path("examples/tutorials/") / notebook
        if list_name in not_excluded_notebooks:
            not_excluded_notebooks.remove(list_name)
            res = pm.execute_notebook(notebook, "/dev/null", parameters={"epochs": 1}, timeout=300)
            assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_fl_with_trainconfig(isolated_filesystem, start_remote_server_worker_only, hook):
    os.chdir("advanced/Federated Learning with TrainConfig/")
    notebook = "Introduction to TrainConfig.ipynb"
    p_name = Path("examples/tutorials/advanced/Federated Learning with TrainConfig/")
    not_excluded_notebooks.remove(p_name / notebook)
    hook.local_worker.remove_worker_from_registry("alice")
    kwargs = {"id": "alice", "host": "localhost", "port": 8777, "hook": hook}
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)
    dataset = sy.BaseDataset(data, target)
    process_remote_worker = start_remote_server_worker_only(dataset=(dataset, "xor"), **kwargs)
    res = pm.execute_notebook(notebook, "/dev/null", timeout=300)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
    process_remote_worker.terminate()
    sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)


@pytest.mark.skip
def test_fl_sms(isolated_filesystem):  # pragma: no cover
    sys.path.append("advanced/Federated SMS Spam prediction/")
    import preprocess

    os.chdir("advanced/Federated SMS Spam prediction/")

    notebook = "Federated SMS Spam prediction.ipynb"
    p_name = Path("examples/tutorials/advanced/Federated SMS Spam prediction/")
    not_excluded_notebooks.remove(p_name / notebook)
    Path("data").mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    urllib.request.urlretrieve(url, "data.zip")
    with ZipFile("data.zip", "r") as zipObj:
        # Extract all the contents of the zip file in current directory
        zipObj.extractall()
    preprocess.main()
    res = pm.execute_notebook(notebook, "/dev/null", parameters={"epochs": 1}, timeout=300)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_fl_with_websockets_and_averaging(
    isolated_filesystem, start_remote_server_worker_only, hook
):
    os.chdir("advanced/websockets-example-MNIST/")
    notebook = "Federated learning with websockets and federated averaging.ipynb"
    p_name = Path("examples/tutorials/advanced/websockets-example-MNIST/")
    not_excluded_notebooks.remove(p_name / notebook)
    for n in ["alice", "bob", "charlie"]:
        hook.local_worker.remove_worker_from_registry(n)
    kwargs_list = [
        {"id": "alice", "host": "localhost", "port": 8777, "hook": hook},
        {"id": "bob", "host": "localhost", "port": 8778, "hook": hook},
        {"id": "charlie", "host": "localhost", "port": 8779, "hook": hook},
    ]
    processes = [start_remote_server_worker_only(**kwargs) for kwargs in kwargs_list]
    res = pm.execute_notebook(
        notebook,
        "/dev/null",
        parameters={"args": ["--epochs", "1", "--test_batch_size", "100"], "abort_after_one": True},
        timeout=300,
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
    [server.terminate() for server in processes]
    for n in ["alice", "bob", "charlie"]:
        sy.VirtualWorker(id=n, hook=hook, is_client_worker=False)


def test_not_tested_notebooks():
    """This test must always be last"""
    assert len(not_excluded_notebooks) == 0, not_excluded_notebooks
