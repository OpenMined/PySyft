import torch
from torch import nn
import syft as sy


def test_copy_on_remote():
    """
    Tests the correctness of copying a remote model.
    """
    data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.0]])
    model = nn.Linear(2, 1)

    pred_local = model(data)

    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    data_bob = data.send(bob)
    model_bob = model.send(bob)
    model_bob_copy = model_bob.copy()
    pred_remote = model_bob_copy(data_bob)
    pred_remote = pred_remote.get()

    assert pred_local.equal(pred_remote)
