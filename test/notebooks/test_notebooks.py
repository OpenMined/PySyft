import glob
import os
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import nbformat
import numpy as np
import pandas as pd
import papermill as pm
import torch

import syft as sy
from syft import TorchHook
from syft.workers.websocket_server import WebsocketServerWorker

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
    """Test Notebooks in the tutorial root folder."""
    notebooks = glob.glob("*.ipynb")
    for notebook in notebooks:
        if Path(notebook).name in exclusion_list_basic:
            continue
        res = pm.execute_notebook(
            notebook, "/dev/null", parameters={"epochs": 1, "n_test_batches": 5}, timeout=300
        )
        assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_notebooks_advanced(isolated_filesystem):
    notebooks = glob.glob("advanced/*.ipynb")
    for notebook in notebooks:
        if Path(notebook).name in exclusion_list_advanced:
            continue
        res = pm.execute_notebook(notebook, "/dev/null", parameters={"epochs": 1}, timeout=300)
        assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_fl_with_trainconfig(isolated_filesystem, start_remote_server_worker_only, hook):
    os.chdir("advanced/Federated Learning with TrainConfig/")
    hook.local_worker.remove_worker_from_registry("alice")
    kwargs = {"id": "alice", "host": "localhost", "port": 8777, "hook": hook}
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)
    dataset = sy.BaseDataset(data, target)
    process_remote_worker = start_remote_server_worker_only(dataset=(dataset, "xor"), **kwargs)
    notebook = "Introduction to TrainConfig.ipynb"
    res = pm.execute_notebook(notebook, "/dev/null", timeout=300)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
    process_remote_worker.terminate()
    sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)


def test_fl_sms(isolated_filesystem):
    os.chdir("advanced/Federated SMS Spam prediction/")
    Path("data").mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    urllib.request.urlretrieve(url, "data.zip")
    with ZipFile("data.zip", "r") as zipObj:
        # Extract all the contents of the zip file in current directory
        zipObj.extractall()
    import preprocess

    preprocess.main()
    res = pm.execute_notebook(
        "Federated SMS Spam prediction.ipynb", "/dev/null", parameters={"epochs": 1}, timeout=300
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_fl_with_websockets_and_averaging(
    isolated_filesystem, start_remote_server_worker_only, hook
):
    os.chdir("advanced/websockets-example-MNIST/")
    for n in ["alice", "bob", "charlie"]:
        hook.local_worker.remove_worker_from_registry(n)
    kwargs_list = [
        {"id": "alice", "host": "localhost", "port": 8777, "hook": hook},
        {"id": "bob", "host": "localhost", "port": 8778, "hook": hook},
        {"id": "charlie", "host": "localhost", "port": 8779, "hook": hook},
    ]
    processes = [start_remote_server_worker_only(**kwargs) for kwargs in kwargs_list]
    notebook = "Federated learning with websockets and federated averaging.ipynb"
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
