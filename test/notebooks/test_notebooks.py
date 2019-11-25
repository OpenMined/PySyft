import papermill as pm
import nbformat
import glob
from pathlib import Path
import torch
import syft as sy
import os
from syft.workers.websocket_server import WebsocketServerWorker
import urllib.request
from zipfile import ZipFile
import pandas as pd
import numpy as np

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


def test_fl_with_trainconfig(isolated_filesystem, start_proc, hook):
    os.chdir("advanced/Federated Learning with TrainConfig/")
    kwargs = {"id": "peter", "host": "localhost", "port": 8777, "hook": hook}
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)
    dataset = sy.BaseDataset(data, target)
    process_remote_worker = start_proc(WebsocketServerWorker, dataset=(dataset, "xor"), **kwargs)
    notebook = "Introduction to TrainConfig.ipynb"
    res = pm.execute_notebook(notebook, "/dev/null", timeout=300)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
    process_remote_worker.terminate()


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
