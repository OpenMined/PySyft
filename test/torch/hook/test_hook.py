import pytest
import torch
import syft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_to():  # pragma: no cover
    a = torch.Tensor([1.0, 2.0, 3.0])
    assert a.is_cuda is False
    a = a.to(torch.device("cuda"))
    assert a.is_cuda is True
    a = a.to(torch.device("cpu"))
    assert a.is_cuda is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_cuda():  # pragma: no cover
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(2, 3)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            return x

    model = Net()
    assert model.fc1.weight.is_cuda is False
    model = model.cuda()
    assert model.fc1.weight.is_cuda is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_data():  # pragma: no cover
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(2, 3)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            return x

    model = Net()
    input = torch.tensor([2.0, 4.0])
    out_cpu = model(input)
    assert model.fc1.weight.is_cuda is False
    model = model.cuda()
    assert model.fc1.weight.is_cuda is True
    out_cuda = model(input.cuda())
    assert (out_cpu - out_cuda.cpu() < 1e-3).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_param_data():  # pragma: no cover
    param = torch.nn.Parameter(data=torch.Tensor([2.0, 3.0]))
    data2 = torch.Tensor([4.0, 5.0]).to("cuda")
    param.data = data2
    assert (param.data == data2).all()
    assert param.is_cuda


def test_send_frozen():
    hook = syft.TorchHook(torch)
    worker = syft.VirtualWorker(hook, id="worker")

    d_in, h, d_out = 1000, 100, 10

    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, h), torch.nn.ReLU(), torch.nn.Linear(h, d_out)
    )

    for param in model.parameters():
        param.requires_grad = False

    model.send(worker)


def test_send_partially_frozen():
    hook = syft.TorchHook(torch)
    worker = syft.VirtualWorker(hook, id="worker")

    d_in, h1, h2, d_out = 1000, 1000, 100, 10

    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, h1),
        torch.nn.ReLU(),
        torch.nn.Linear(h1, h2),
        torch.nn.ReLU(),
        torch.nn.Linear(h2, d_out),
    )

    for layer_idx, param in enumerate(model.parameters()):
        if layer_idx > 2:  # freezing the first two layers
            pass
        param.requires_grad = False

    model.send(worker)
