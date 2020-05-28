import importlib

from syft.frameworks.torch.hook.hook import TorchHook
from syft.workers.virtual import VirtualWorker
from syft.grid.private_grid import PrivateGridNetwork

from syft.exceptions import DependencyError


def create_sandbox(gbs, verbose=True, download_data=True):  # noqa: C901
    """There's some boilerplate stuff that most people who are
    just playing around would like to have. This will create
    that for you"""

    try:
        torch = gbs["torch"]
    except KeyError:
        torch = gbs["th"]

    global hook
    global bob
    global theo
    global alice
    global andy
    global jason
    global jon

    if download_data and importlib.util.find_spec("sklearn") is None:
        raise DependencyError("sklearn", "scikit-learn")

    if download_data:  # pragma: no cover
        from sklearn.datasets import load_boston
        from sklearn.datasets import load_breast_cancer
        from sklearn.datasets import load_digits
        from sklearn.datasets import load_diabetes
        from sklearn.datasets import load_iris
        from sklearn.datasets import load_wine
        from sklearn.datasets import load_linnerud

        def load_sklearn(func, *tags):
            dataset = func()
            data = (
                torch.tensor(dataset["data"])
                .float()
                .tag(*(list(tags) + ["#data"] + dataset["DESCR"].split("\n")[0].lower().split(" ")))
                .describe(dataset["DESCR"])
            )
            target = (
                torch.tensor(dataset["target"])
                .float()
                .tag(
                    *(list(tags) + ["#target"] + dataset["DESCR"].split("\n")[0].lower().split(" "))
                )
                .describe(dataset["DESCR"])
            )

            return data, target

        def distribute_dataset(data, workers):
            batch_size = int(data.shape[0] / len(workers))
            n_batches = len(workers)
            for batch_i in range(n_batches - 1):
                batch = data[batch_i * batch_size : (batch_i + 1) * batch_size]
                batch.tags = data.tags
                batch.description = data.description
                ptr = batch.send(workers[batch_i])
                ptr.child.garbage_collect_data = False

            batch = data[(n_batches - 1) * batch_size :]
            batch.tags = data.tags
            batch.description = data.description
            ptr = batch.send(workers[n_batches - 1])
            ptr.child.garbage_collect_data = False

    print("Setting up Sandbox...")

    if verbose:
        print("\t- Hooking PyTorch")
    hook = TorchHook(torch)

    if verbose:
        print("\t- Creating Virtual Workers:")
        print("\t\t- bob")
    bob = VirtualWorker(hook, id="bob")
    if verbose:
        print("\t\t- theo")
    theo = VirtualWorker(hook, id="theo")
    if verbose:
        print("\t\t- jason")
    jason = VirtualWorker(hook, id="jason")
    if verbose:
        print("\t\t- alice")
    alice = VirtualWorker(hook, id="alice")
    if verbose:
        print("\t\t- andy")
    andy = VirtualWorker(hook, id="andy")
    if verbose:
        print("\t\t- jon")
    jon = VirtualWorker(hook, id="jon")

    if verbose:
        print("\tStoring hook and workers as global variables...")
    gbs["hook"] = hook
    gbs["bob"] = bob
    gbs["theo"] = theo
    gbs["jason"] = jason
    gbs["alice"] = alice
    gbs["andy"] = andy
    gbs["jon"] = jon

    gbs["workers"] = [bob, theo, jason, alice, andy, jon]

    if download_data:  # pragma: no cover

        if verbose:
            print("\tLoading datasets from SciKit Learn...")
            print("\t\t- Boston Housing Dataset")
        boston = load_sklearn(load_boston, *["#boston", "#housing", "#boston_housing"])
        if verbose:
            print("\t\t- Diabetes Dataset")
        diabetes = load_sklearn(load_diabetes, *["#diabetes"])
        if verbose:
            print("\t\t- Breast Cancer Dataset")
        breast_cancer = load_sklearn(load_breast_cancer)
        if verbose:
            print("\t- Digits Dataset")
        digits = load_sklearn(load_digits)
        if verbose:
            print("\t\t- Iris Dataset")
        iris = load_sklearn(load_iris)
        if verbose:
            print("\t\t- Wine Dataset")
        wine = load_sklearn(load_wine)
        if verbose:
            print("\t\t- Linnerud Dataset")
        linnerud = load_sklearn(load_linnerud)

        workers = [bob, theo, jason, alice, andy, jon]

        if verbose:
            print("\tDistributing Datasets Amongst Workers...")
        distribute_dataset(boston[0], workers)
        distribute_dataset(boston[1], workers)
        distribute_dataset(diabetes[0], workers)
        distribute_dataset(diabetes[1], workers)
        distribute_dataset(breast_cancer[0], workers)
        distribute_dataset(breast_cancer[1], workers)
        distribute_dataset(digits[0], workers)
        distribute_dataset(digits[1], workers)
        distribute_dataset(iris[0], workers)
        distribute_dataset(iris[1], workers)
        distribute_dataset(wine[0], workers)
        distribute_dataset(wine[1], workers)
        distribute_dataset(linnerud[0], workers)
        distribute_dataset(linnerud[1], workers)

    if verbose:
        print("\tCollecting workers into a VirtualGrid...")
    _grid = PrivateGridNetwork(*gbs["workers"])
    gbs["grid"] = _grid

    print("Done!")


def make_hook(gbs):
    return create_sandbox(gbs, False, False)
