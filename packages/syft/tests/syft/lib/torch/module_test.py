# stdlib
import copy
import os
from pathlib import Path
import time
from typing import Any
from typing import Tuple

# third party
import pytest
import torch
import torch as th

# syft absolute
import syft as sy
from syft import SyModule
from syft import SySequential
from syft.core.plan.plan import Plan
from syft.core.plan.plan_builder import ROOT_CLIENT
from syft.core.plan.plan_builder import make_plan

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


class SyNetEmpty(sy.Module):
    """
    Simple test model
    """

    def __init__(self) -> None:
        super(SyNetEmpty, self).__init__(torch_ref=torch)

    def forward(self, x: torch.Tensor) -> Any:
        return 0


class MySyModuleBlock(SyModule):
    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        self.p1 = th.nn.Parameter(th.rand(100, 10) * 0.01)

    def forward(self, x):  # type: ignore
        o1 = x @ self.p1
        return o1


class MySyModule(SyModule):
    def __init__(self, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        self.layer1 = th.nn.Linear(28 * 28, 100)
        self.relu1 = th.nn.ReLU()
        self.layer2 = MySyModuleBlock(input_size=(32, 100))

    def forward(self, x):  # type: ignore
        x_reshaped = x.view(-1, 28 * 28)
        o1 = self.layer1(x_reshaped)
        a1 = self.relu1(o1)
        out = self.layer2(x=a1)[0]
        return out


class MyTorchModuleBlock(th.nn.Module):
    def __init__(self):  # type: ignore
        super().__init__()
        self.p1 = th.nn.Parameter(th.rand(100, 10) * 0.01)

    def forward(self, x):  # type: ignore
        o1 = x @ self.p1
        return o1


class MyTorchModule(th.nn.Module):
    def __init__(self):  # type: ignore
        super().__init__()
        self.layer1 = th.nn.Linear(28 * 28, 100)
        self.relu1 = th.nn.ReLU()
        self.layer2 = MyTorchModuleBlock()

    def forward(self, x):  # type: ignore
        x_reshaped = x.view(-1, 28 * 28)
        o1 = self.layer1(x_reshaped)
        a1 = self.relu1(o1)
        out = self.layer2(a1)
        return out


class MySySequentialBlock(SyModule):
    def __init__(self, n_in, n_out, **kwargs):  # type: ignore
        super().__init__(**kwargs)
        self.layer = th.nn.Linear(n_in, n_out)

    def forward(self, x):  # type: ignore
        o1 = self.layer(x)
        return o1


@pytest.fixture(scope="function")
def model() -> SyNet:
    return SyNet()


@pytest.fixture(scope="function")
def modelEmpty() -> SyNetEmpty:
    return SyNetEmpty()


@pytest.fixture(scope="function")
def dataloader() -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(size=(1, IN_DIM)), torch.randn(size=(1, OUT_DIM))


@pytest.fixture(scope="function")
def sy_model() -> SyNet:
    return MySyModule(input_size=(32, 28 * 28))  # type: ignore


@pytest.fixture(scope="function")
def torch_model() -> SyNet:
    return MyTorchModule()  # type: ignore


@pytest.fixture(scope="function")
def sy_sequential() -> SyNet:
    return SySequential(
        MySySequentialBlock(100, 10, input_size=(32, 100)),  # type: ignore
        MySySequentialBlock(10, 10, input_size=(32, 10)),  # type: ignore
    )


def test_repr_to_kwargs() -> None:
    assert sy.lib.util.full_name_with_qualname(klass=torch.Tensor) == "torch.Tensor"
    assert sy.lib.torch.module.repr_to_kwargs(
        "1, 32, kernel_size=(3, 3), stride=(1, 1)"
    ) == ([1, 32], {"kernel_size": (3, 3), "stride": (1, 1)})
    assert sy.lib.torch.module.repr_to_kwargs("1, 32") == ([1, 32], {})
    assert sy.lib.torch.module.repr_to_kwargs("kernel_size=(3, 3), stride=(1, 1)") == (
        [],
        {"kernel_size": (3, 3), "stride": (1, 1)},
    )


def test_module_setup(root_client: sy.VirtualMachineClient, model: SyNet) -> None:
    remote = copy.copy(model)
    remote.setup(torch_ref=root_client.torch)
    assert remote.is_local is False
    assert remote.torch_ref == root_client.torch
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


def test_module_modules_empty(modelEmpty: SyNetEmpty) -> None:
    modules = modelEmpty.modules
    assert len(modules.items()) == 0


@pytest.mark.slow
def test_module_parameteres(root_client: sy.VirtualMachineClient, model: SyNet) -> None:
    model_ptr = model.send(root_client)

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

    new_model.is_local = False

    assert new_model.load_state_dict(state) is None
    assert new_model.state_dict() is None


def test_module_load_save(model: SyNet) -> None:
    state = model.state_dict()

    folder = Path("tmp")
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    path = folder / str(time.time())
    model.save(path)

    model.is_local = False
    assert model.save(path) is None

    new_model = SyNet()
    new_model.load(path)
    new_state = new_model.state_dict()

    new_model.is_local = False
    assert new_model.load(path) is None

    try:
        os.remove(path)
    except BaseException:
        pass

    for k in state:
        assert k in new_state
        assert torch.all(torch.eq(new_state[k], state[k]))


def test_module_gradient_sanity(
    model: SyNet,
    dataloader: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    data, labels = dataloader

    result = model(data)
    loss_func = torch.nn.L1Loss()
    loss = loss_func(result, labels)
    loss.backward()

    assert model.parameters()[-1].grad is not None


@pytest.mark.slow
def test_module_send_get(
    root_client: sy.VirtualMachineClient,
    model: SyNet,
    dataloader: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    data, labels = dataloader

    model_ptr = model.send(root_client)
    data_ptr = data.send(root_client)
    labels_ptr = labels.send(root_client)

    results_ptr = model_ptr(data_ptr)
    remote_loss_func = root_client.torch.nn.L1Loss()
    remote_loss = remote_loss_func(results_ptr, labels_ptr)
    remote_loss.backward()

    direct_param = model_ptr.parameters().get()
    for param in direct_param:
        assert param.grad is not None

    # get() uses state_dict/load_state_dict
    # load_state_dict breaks the computational graph, and we won't have the gradients here.
    # ref: https://discuss.pytorch.org/t/loading-a-state-dict-seems-to-erase-grad/56676
    model_parameter = model_ptr.get().parameters()
    for param in model_parameter:
        assert param.grad is None

    for idx, param in enumerate(direct_param):
        assert param.tolist() == model_parameter[idx].tolist()

    assert model.get() is None

    model.is_local = False
    assert model.send(root_client) is None


@pytest.mark.slow
def test_debug_sum_layers(root_client: sy.VirtualMachineClient, model: SyNet) -> None:
    assert model.debug_sum_layers() is None
    model_ptr = model.send(root_client)

    assert model_ptr.debug_sum_layers() is None


def test_sy_module(
    root_client: sy.VirtualMachineClient,
    sy_model: SyModule,
    torch_model: torch.nn.Module,
) -> None:
    assert isinstance(sy_model._forward_plan, Plan)
    assert len(sy_model._forward_plan.actions) > 0
    assert sy_model.state_dict().keys() == torch_model.state_dict().keys()

    sy_model.load_state_dict(torch_model.state_dict())
    sy_model_ptr = sy_model.send(ROOT_CLIENT)

    x = th.randn(32, 28 * 28)

    sy_out = sy_model(x=x)[0]
    sy_ptr_out = sy_model_ptr(x=x).get()[0]
    torch_out = torch_model(x)
    assert th.equal(torch_out, sy_ptr_out)
    assert th.equal(torch_out, sy_out)


@pytest.mark.slow
def test_recompile_downloaded_sy_module(
    sy_model: SyModule,
    torch_model: torch.nn.Module,
) -> None:
    # first download
    downloaded_sy_model = sy_model.send(ROOT_CLIENT).get()
    # then load new weights
    downloaded_sy_model.load_state_dict(torch_model.state_dict())
    # then execute & compare
    x = th.randn(32, 28 * 28)
    sy_out = downloaded_sy_model(x=x)[0]
    torch_out = torch_model(x)
    assert th.equal(torch_out, sy_out)


@pytest.mark.slow
def test_nest_sy_module(
    root_client: sy.VirtualMachineClient, sy_model: SyModule
) -> None:
    remote_torch = ROOT_CLIENT.torch

    @make_plan
    def train(model=sy_model):  # type: ignore
        optimizer = remote_torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        out = model(x=th.randn(32, 28 * 28))[0]
        loss = remote_torch.nn.functional.cross_entropy(out, th.randint(10, (32,)))
        loss.backward()
        optimizer.step()
        return [model]

    (new_model,) = train(model=sy_model)
    assert not th.equal(
        sy_model.state_dict()["layer1.weight"], new_model.state_dict()["layer1.weight"]
    )


def test_sy_sequential(
    root_client: sy.VirtualMachineClient, sy_sequential: SySequential
) -> None:
    for module in sy_sequential:
        assert isinstance(module, SyModule)
        assert isinstance(module._forward_plan, Plan)

    (res,) = sy_sequential(x=th.randn(32, 100))
    assert res.shape == (32, 10)
