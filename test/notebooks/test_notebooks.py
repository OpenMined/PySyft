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

thismodule = sys.modules[__name__]

# lets start by finding all notebooks currently available in examples and subfolders
all_notebooks = [n for n in glob.glob("examples/tutorials/**/*.ipynb", recursive=True)]
basic_notebooks = [n for n in glob.glob("examples/tutorials/*.ipynb")]
advanced_notebooks = [
    n for n in glob.glob("examples/tutorials/advanced/**/*.ipynb", recursive=True)
]
translated_notebooks = [
    n for n in glob.glob("examples/tutorials/translations/**/*.ipynb", recursive=True)
]
# Exclude all translated basic tutorials that are also
# excluded in their original version.
excluded_translated_notebooks = [
    Path(nb).name for part in ["10", "13b", "13c"] for nb in translated_notebooks if part in nb
]

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
    # Outdated training method
    "Introduction to TrainConfig.ipynb",
    # Outdated websocket client code
    "Federated learning with websockets and federated averaging.ipynb",
]

# Add excluded translated notebooks to the exclusion list
exclusion_list_notebooks += excluded_translated_notebooks

exclusion_list_folders = [
    "examples/tutorials/websocket",
    "examples/tutorials/advanced/Monitor_Network_Traffic",
    "examples/tutorials/advanced/websockets-example-MNIST-parallel",
    # To run these notebooks, we need to run grid nodes / grid gateway previously (they aren't  in this repository)
    "examples/tutorials/grid",
    "examples/tutorials/grid/federated_learning/spam_prediction",
    "examples/tutorials/grid/federated_learning/mnist",
    # This notebook is skipped because it fails in github actions and we do not know why for the moment
    "examples/tutorials/advanced/Federated SMS Spam prediction",
]


excluded_notebooks = []
for nb in all_notebooks:
    if Path(nb).name in exclusion_list_notebooks:
        excluded_notebooks += [nb]
for folder in exclusion_list_folders:
    excluded_notebooks += glob.glob(f"{folder}/**/*.ipynb", recursive=True)

tested_notebooks = []


@pytest.mark.parametrize("notebook", sorted(set(basic_notebooks) - set(excluded_notebooks)))
def test_notebooks_basic(isolated_filesystem, notebook):
    """Test Notebooks in the tutorial root folder."""
    notebook = notebook.split("/")[-1]
    list_name = Path("examples/tutorials/") / notebook
    tested_notebooks.append(str(list_name))
    res = pm.execute_notebook(
        notebook,
        "/dev/null",
        parameters={"epochs": 1, "n_test_batches": 5, "n_train_items": 64, "n_test_items": 64},
        timeout=300,
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


for language_dir in glob.glob("examples/tutorials/translations/*"):
    language = Path(language_dir).name
    language_notebooks = [
        n for n in glob.glob(f"examples/tutorials/translations/{language}/*.ipynb", recursive=True)
    ]

    def language_test_helper(language):
        @pytest.mark.translation
        @pytest.mark.parametrize(
            "language_notebook", sorted(set(language_notebooks) - set(excluded_notebooks))
        )
        def test_basic_notebook_translations_per_language(
            isolated_filesystem, language_notebook
        ):  # pragma: no cover
            """Test Notebooks in the tutorial translations folder."""
            notebook = "/".join(language_notebook.split("/")[-2:])
            notebook = f"translations/{notebook}"
            list_name = Path(f"examples/tutorials/{notebook}")
            tested_notebooks.append(str(list_name))
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

        return test_basic_notebook_translations_per_language

    setattr(
        thismodule,
        f"test_basic_notebook_translations_in_{language}",
        language_test_helper(language),
    )


@pytest.mark.parametrize("notebook", sorted(set(advanced_notebooks) - set(excluded_notebooks)))
def test_notebooks_advanced(isolated_filesystem, notebook):
    notebook = notebook.replace("examples/tutorials/", "")
    list_name = Path("examples/tutorials/") / notebook
    tested_notebooks.append(str(list_name))
    res = pm.execute_notebook(notebook, "/dev/null", parameters={"epochs": 1}, timeout=300)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


def test_fl_with_trainconfig(isolated_filesystem, start_remote_server_worker_only, hook):
    os.chdir("advanced/Federated Learning with TrainConfig/")
    notebook = "Introduction to TrainConfig.ipynb"
    p_name = Path("examples/tutorials/advanced/Federated Learning with TrainConfig/")
    tested_notebooks.append(str(p_name / notebook))
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
    tested_notebooks.append(str(p_name / notebook))
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
    tested_notebooks.append(str(p_name / notebook))
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


def test_github_workflows_exist_for_all_languages():
    dirs_checked = 0
    # check that all languages in translation directory have a workflow
    for language_dir in glob.glob("examples/tutorials/translations/*"):
        language = Path(language_dir).name
        workflow = Path(".github", "workflows", f"tutorials-{language}.yml")
        assert os.path.exists(workflow) and os.path.isfile(workflow)
        dirs_checked += 1
    assert dirs_checked > 0


### This test must always be last
def test_all_notebooks_except_translations():
    untested_notebooks = (
        set(all_notebooks)
        - set(excluded_notebooks)
        - set(translated_notebooks)
        - set(tested_notebooks)
    )
    assert len(untested_notebooks) == 0, untested_notebooks
