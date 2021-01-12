# third party
import torch
import pytest
from typing import Any, Tuple

# syft absolute
import syft as sy

IN_DIM = 100
OUT_DIM = 10


class SyNet(sy.Module):
    """
    Simple test model
    """

    def __init__(self) -> None:
        super(SyNet, self).__init__(torch_ref=torch)
        self.fc1 = torch.nn.Linear(IN_DIM, OUT_DIM)

    def forward(self, x: torch.Tensor) -> Any:
        return self.fc1(x)


@pytest.fixture(scope="function")
def alice() -> sy.VirtualMachine:
    return sy.VirtualMachine(name="alice")


@pytest.fixture(scope="function")
def model() -> SyNet:
    return SyNet()


@pytest.fixture(scope="function")
def dataloader() -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(size=(1, IN_DIM)), torch.randn(size=(1, OUT_DIM))


def test_module_gradient_sanity(
    alice: sy.VirtualMachine,
    model: SyNet,
    dataloader: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    data, labels = dataloader

    result = model(data)
    loss_func = torch.nn.L1Loss()
    loss = loss_func(result, labels)
    loss.backward()

    assert model.parameters()[-1].grad is not None


def test_module_remote_sanity(
    alice: sy.VirtualMachine,
    model: SyNet,
    dataloader: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    alice_client = alice.get_root_client()

    data, labels = dataloader

    model_ptr = model.send(alice_client)
    data_ptr = data.send(alice_client)
    labels_ptr = labels.send(alice_client)
    results_ptr = model_ptr(data_ptr)
    remote_loss_func = alice_client.torch.nn.L1Loss()
    remote_loss = remote_loss_func(results_ptr, labels_ptr)
    remote_loss.backward()

    direct_param = model_ptr.parameters().get()
    model_parameter = model_ptr.get().parameters()

    for idx, param in enumerate(direct_param):
        assert param.tolist() == model_parameter[idx].tolist()
        assert param.grad == model_parameter[idx].grad
