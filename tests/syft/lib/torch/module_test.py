# stdlib
import copy
import os
import time
from pathlib import Path

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


def test_repr_to_kwargs() -> None:
    assert sy.lib.util.full_name_with_qualname(klass=torch.Tensor) == "torch.Tensor"


def test_module_setup(alice: sy.VirtualMachine, model: SyNet) -> None:
    alice_client = alice.get_root_client()

    remote = copy.copy(model)
    remote.setup(torch_ref=alice_client.torch)
    assert remote.is_local is False
    assert remote.torch_ref == alice_client.torch
    assert remote.training is False

    remote.setup(torch_ref=torch)
    assert remote.is_local is True


def test_module_attr(model: SyNet) -> None:
    model.__setattr__("fc1", torch.nn.Linear(1, 2))
    assert model.__getattr__("fc1").in_features == 1
    assert model.__getattr__("fc1").out_features == 2


def test_module_modules(model: SyNet) -> None:
    modules = model.modules
    assert len(modules.items()) == 1
    assert "fc1" in modules
    assert modules["fc1"].in_features == IN_DIM


def test_module_parameteres(alice: sy.VirtualMachine, model: SyNet) -> None:
    model_ptr = model.send(alice.get_root_client())

    assert len(model_ptr.parameters().get()) == 2
    assert model_ptr.parameters().get()[0].shape == torch.Size([OUT_DIM, IN_DIM])
    assert model_ptr.parameters().get()[1].shape == torch.Size([OUT_DIM])


def test_module_cuda(model: SyNet) -> None:
    model.cpu()
    assert model.parameters()[-1].is_cuda is False


def test_module_zero(model: SyNet) -> None:
    model.zero_layers()
    for _, m in model.modules.items():
        for _, v in m.state_dict().items():
            assert v.sum() == 0


def test_module_state_dict(model: SyNet) -> None:
    state = model.state_dict()

    new_model = SyNet()
    new_model.load_state_dict(state)

    new_state = new_model.state_dict()
    for k in state:
        assert k in new_state
        assert torch.all(torch.eq(new_state[k], state[k]))


def test_module_load_save(model: SyNet) -> None:
    state = model.state_dict()

    folder = Path("tmp")
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    path = folder / str(time.time())
    model.save(path)

    new_model = SyNet()
    new_model.load(path)
    new_state = new_model.state_dict()

    try:
        os.remove(path)
    except BaseException:
        pass

    for k in state:
        assert k in new_state
        assert torch.all(torch.eq(new_state[k], state[k]))


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


def test_module_send_get(
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
        assert param.grad is not None
        assert model_parameter[idx].grad is not None
        assert param.tolist() == model_parameter[idx].tolist()
        assert param.grad == model_parameter[idx].grad
