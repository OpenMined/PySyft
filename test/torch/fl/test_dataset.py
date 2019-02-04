import pytest
import torch as th
import syft as sy


def test_federated_dataset():
    sy.create_sandbox(globals(), verbose=False)

    boston_data, _ = grid.search("#boston", "#data", verbose=False)
    boston_target, _ = grid.search("#boston", "#target", verbose=False)

    n_inputs = boston_data['alice'][0].shape[1]
    n_targets = 1

    model = th.nn.Linear(n_inputs,n_targets, bias=False)
    opt = sy.optim.SGD(params=model.parameters(),lr=0.0000001)

    fd = sy.FederatedDataset(boston_data, boston_target, batch_size=32, num_iterators=1)

    for iter in range(100):
        fd.reset()
        i = 0
        loss_accum = 0
        while(fd.keep_going()):
            i += 1

            data, target = fd.step()
            