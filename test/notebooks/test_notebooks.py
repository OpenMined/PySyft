import glob
import os
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile
import codecs

import pytest
import nbformat
import papermill as pm

import syft as sy

# lets start by finding all notebooks currently available in examples and subfolders
all_notebooks = (n for n in glob.glob("examples/tutorials/**/*.ipynb", recursive=True))
basic_notebooks = (n for n in glob.glob("examples/tutorials/*.ipynb"))
advanced_notebooks = (
    n for n in glob.glob("examples/tutorials/advanced/**/*.ipynb", recursive=True)
)
translated_notebooks = (
    n for n in glob.glob("examples/tutorials/translations/**/*.ipynb", recursive=True)
)
# Exclude all translated basic tutorials that are also
# excluded in their original version.
excluded_translated_notebooks = [
    Path(nb).name for part in ["10", "13b", "13c"] for nb in translated_notebooks if part in nb
]


# Include only the translations that have been changed
gitdiff = Path("test/notebooks/git-diff.txt")
changed_files = []
if gitdiff.is_file():
    changed_files = open(gitdiff, "r")
    changed_files = changed_files.readlines()
    changed_files = [
        codecs.decode(file.replace('"', "").replace("\n", ""), "unicode-escape")
        .encode("latin-1")
        .decode()
        for file in changed_files
    ]
translated_notebooks_diff = list(set(changed_files) & set(translated_notebooks))

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
    # Outdated websocket client code
    "Federated learning with websockets and federated averaging.ipynb",
]

# Add excluded translated notebooks to the exclusion list
exclusion_list_notebooks += excluded_translated_notebooks

exclusion_list_folders = [
    "examples/tutorials/websocket",
    "examples/tutorials/advanced/monitor_network_traffic",
    "examples/tutorials/advanced/privacy_attacks",
    "examples/tutorials/advanced/websockets_mnist_parallel",
    # To run these notebooks, we need to run grid nodes / grid gateway
    # previously (they aren't  in this repository)
    "examples/tutorials/grid",
    "examples/tutorials/grid/federated_learning/spam_prediction",
    "examples/tutorials/grid/federated_learning/mnist",
    # This notebook is skipped because it fails in github actions and we
    # do not know why for the moment
    "examples/tutorials/advanced/federated_sms_spam_prediction",
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


@pytest.mark.translation
@pytest.mark.parametrize(
    "translated_notebook", sorted(set(translated_notebooks) - set(excluded_notebooks))
)
def test_notebooks_basic_translations(isolated_filesystem, translated_notebook):  # pragma: no cover
    """Test Notebooks in the tutorial translations folder."""
    notebook = "/".join(translated_notebook.split("/")[-2:])
    notebook = f"translations/{notebook}"
    list_name = Path(f"examples/tutorials/{notebook}")
    tested_notebooks.append(str(list_name))
    res = pm.execute_notebook(
        notebook,
        "/dev/null",
        parameters={"epochs": 1, "n_test_batches": 5, "n_train_items": 64, "n_test_items": 64},
        timeout=400,
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


@pytest.mark.translation
@pytest.mark.parametrize(
    "translated_notebook", sorted(set(translated_notebooks_diff) - set(excluded_notebooks))
)
def test_notebooks_basic_translations_diff(
    isolated_filesystem, translated_notebook
):  # pragma: no cover
    """
    Test Notebooks in the tutorial translations folder if they have been
    modified in the current pull request. This test should not consider any
    notebooks locally. It should be used on Github Actions.
    """
    notebook = "/".join(translated_notebook.split("/")[-2:])
    notebook = f"translations/{notebook}"
    list_name = Path(f"examples/tutorials/{notebook}")
    tested_notebooks.append(str(list_name))
    res = pm.execute_notebook(
        notebook,
        "/dev/null",
        parameters={"epochs": 1, "n_test_batches": 5, "n_train_items": 64, "n_test_items": 64},
        timeout=300,
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


@pytest.mark.parametrize("notebook", sorted(set(advanced_notebooks) - set(excluded_notebooks)))
def test_notebooks_advanced(isolated_filesystem, notebook):
    notebook = notebook.replace("examples/tutorials/", "")
    list_name = Path("examples/tutorials/") / notebook
    tested_notebooks.append(str(list_name))
    res = pm.execute_notebook(notebook, "/dev/null", parameters={"epochs": 1}, timeout=400)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)


@pytest.mark.skip
def test_fl_sms(isolated_filesystem):  # pragma: no cover
    sys.path.append("advanced/federated_sms_spam_prediction/")
    import preprocess

    os.chdir("advanced/federated_sms_spam_prediction/")

    notebook = "Federated SMS Spam prediction.ipynb"
    p_name = Path("examples/tutorials/advanced/federated_sms_spam_prediction/")
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
    os.chdir("advanced/websockets_mnist/")
    notebook = "Federated learning with websockets and federated averaging.ipynb"
    p_name = Path("examples/tutorials/advanced/websockets_mnist/")
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


### These tests must always be last
def test_all_notebooks_except_translations():
    untested_notebooks = (
        set(all_notebooks)
        - set(excluded_notebooks)
        - set(translated_notebooks)
        - set(tested_notebooks)
    )
    assert len(untested_notebooks) == 0, untested_notebooks


@pytest.mark.translation
def test_all_translation_notebooks():  # pragma: no cover
    untested_notebooks = set(translated_notebooks) - set(excluded_notebooks) - set(tested_notebooks)
    assert len(untested_notebooks) == 0, untested_notebooks
