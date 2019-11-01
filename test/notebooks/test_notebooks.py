import papermill as pm
import nbformat
import glob


def test_notebooks(isolated_filesystem):
    notebooks = glob.glob("*.ipynb")
    for notebook in notebooks:
        print(notebook)
        res = pm.execute_notebook(
            notebook, "/dev/null", parameters={"epochs": 1, "n_test_batches": 5}, timeout=300
        )
        print(type(res))
        assert isinstance(res, nbformat.notebooknode.NotebookNode)
