import torch as th
import syft as sy

from syft.frameworks.torch.federated import utils
from syft.frameworks.torch import federated


def test_extract_batches_per_worker(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    datasets = [
        federated.BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        federated.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
    ]
    fed_dataset = sy.FederatedDataset(datasets)

    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=2, shuffle=True)

    batches = utils.extract_batches_per_worker(fdataloader)

    assert len(batches.keys()) == len(
        datasets
    ), "each worker should appear as key in the batches dictionary"
