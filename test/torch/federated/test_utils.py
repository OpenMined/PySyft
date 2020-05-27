import pytest

import torch as th
import syft as sy

from syft.frameworks.torch.fl import utils
from syft.frameworks.torch import fl


def test_extract_batches_per_worker(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    datasets = [
        fl.BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        fl.BaseDataset(th.tensor([3, 4, 5, 6]), th.tensor([3, 4, 5, 6])).send(alice),
    ]
    fed_dataset = sy.FederatedDataset(datasets)

    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=2, shuffle=True)

    batches = utils.extract_batches_per_worker(fdataloader)

    assert len(batches.keys()) == len(
        datasets
    ), "each worker should appear as key in the batches dictionary"


def test_add_model():
    class Net(th.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 2)

    weight1 = th.tensor([1.0, 2.0, 3.0, 4.0])
    weight2 = th.tensor([11.0, 22.0, 33.0, 44.0])

    bias1 = th.tensor([-1.0, -2.0])
    bias2 = th.tensor([1.0, 2.0])

    net1 = Net()
    params1 = net1.named_parameters()
    dict_params1 = dict(params1)
    with th.no_grad():
        dict_params1["fc1.weight"].set_(weight1)
        dict_params1["fc1.bias"].set_(bias1)

    net2 = Net()
    params2 = net2.named_parameters()
    dict_params2 = dict(params2)
    with th.no_grad():
        dict_params2["fc1.weight"].set_(weight2)
        dict_params2["fc1.bias"].set_(bias2)

    new_model = utils.add_model(net1, net2)

    assert (new_model.fc1.weight.data == (weight1 + weight2)).all()
    assert (new_model.fc1.bias.data == (bias1 + bias2)).all()


@pytest.mark.skipif(not th.cuda.is_available(), reason="cuda not available")
def test_add_model_cuda():  # pragma: no cover
    class Net(th.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 2)

    weight1 = th.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
    weight2 = th.tensor([11.0, 22.0, 33.0, 44.0]).cuda()

    bias1 = th.tensor([-1.0, -2.0]).cuda()
    bias2 = th.tensor([1.0, 2.0]).cuda()

    net1 = Net().to(th.device("cuda"))
    params1 = net1.named_parameters()
    dict_params1 = dict(params1)
    with th.no_grad():
        dict_params1["fc1.weight"].set_(weight1)
        dict_params1["fc1.bias"].set_(bias1)

    net2 = Net().cuda()
    params2 = net2.named_parameters()
    dict_params2 = dict(params2)
    with th.no_grad():
        dict_params2["fc1.weight"].set_(weight2)
        dict_params2["fc1.bias"].set_(bias2)

    new_model = utils.add_model(net1, net2)

    assert (new_model.fc1.weight.data == (weight1 + weight2)).all()
    assert (new_model.fc1.bias.data == (bias1 + bias2)).all()


def test_scale_model():
    class Net(th.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 2)

    weight1 = th.tensor([1.0, 2.0, 3.0, 4.0])

    bias1 = th.tensor([-1.0, -2.0])

    net1 = Net()
    params1 = net1.named_parameters()
    dict_params1 = dict(params1)
    with th.no_grad():
        dict_params1["fc1.weight"].set_(weight1)
        dict_params1["fc1.bias"].set_(bias1)

    scale = 2.0

    new_model = utils.scale_model(net1, scale)

    assert (new_model.fc1.weight.data == (weight1 * scale)).all()
    assert (new_model.fc1.bias.data == (bias1 * scale)).all()


def test_accuracy():
    pred = th.tensor([[0.95, 0.02, 0.03], [0.3, 0.4, 0.3], [0.0, 0.0, 1.0]])

    target = th.tensor([0.0, 1.0, 2.0])

    acc = utils.accuracy(pred, target)

    assert acc == 1.0

    target = th.tensor([2.0, 0.0, 2.0])

    acc = utils.accuracy(pred, target)

    assert acc == 1.0 / 3.0


def test_federated_avg():
    class Net(th.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = th.nn.Linear(2, 2)

    net1 = Net()
    net2 = Net()
    net3 = Net()

    models = {}
    models[0] = net1
    models[1] = net2
    models[2] = net3

    avg_model = utils.federated_avg(models)
    assert avg_model != net1
    assert (avg_model.fc1.weight.data != net1.fc1.weight.data).all()
    assert (avg_model.fc1.bias.data != net1.fc1.bias.data).all()
