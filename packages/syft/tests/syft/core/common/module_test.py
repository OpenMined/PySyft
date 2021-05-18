# stdlib
from typing import Any
from typing import List
from typing import Union

# third party
import torch as th

# syft absolute
import syft as sy


def test_module() -> None:
    torch = sy.lib.torch.torch
    nn = torch.nn
    F = torch.nn.functional

    # manually constructed Module with PyTorch external API
    class Net:
        modules: List[Any] = []
        training = False

        def __init__(self) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

            # add to modules list
            self.modules.append(self.conv1)
            self.modules.append(self.conv2)
            self.modules.append(self.dropout1)
            self.modules.append(self.dropout2)
            self.modules.append(self.fc1)
            self.modules.append(self.fc2)

        def train(self, mode: bool = True) -> Any:
            self.training = mode
            for module in self.modules:
                module.train(mode)
            return self

        def eval(self) -> Any:
            return self.train(False)

        def forward(self, x: Any) -> Any:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

        def __call__(self, input: Any) -> Any:
            return self.forward(input)

        # local list of remote ListPointers of TensorPointers
        def parameters(self, recurse: bool = True) -> List:
            params_list = torch.python.List()
            for module in self.modules:
                param_pointers = module.parameters()
                params_list += param_pointers

            return params_list

        def cuda(self, device: Any) -> "Net":
            for module in self.modules:
                module.cuda(device)
            return self

        def cpu(self) -> "Net":
            for module in self.modules:
                module.cpu()
            return self

    # sy.Module version with equivalent external interface
    class SyNet(sy.Module):
        def __init__(self, torch_ref: Any) -> None:
            super(SyNet, self).__init__(torch_ref=torch_ref)
            self.conv1 = torch_ref.nn.Conv2d(1, 32, 3, 1)
            self.conv2 = torch_ref.nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = torch_ref.nn.Dropout2d(0.25)
            self.dropout2 = torch_ref.nn.Dropout2d(0.5)
            self.fc1 = torch_ref.nn.Linear(9216, 128)
            self.fc2 = torch_ref.nn.Linear(128, 10)

        def forward(self, x: Any) -> Any:
            x = self.conv1(x)
            x = self.torch_ref.nn.functional.relu(x)
            x = self.conv2(x)
            x = self.torch_ref.nn.functional.relu(x)
            x = self.torch_ref.nn.functional.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = self.torch_ref.flatten(x, 1)
            x = self.fc1(x)
            x = self.torch_ref.nn.functional.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = self.torch_ref.nn.functional.log_softmax(x, dim=1)
            return output

    vanilla_model = Net()
    sy_model = SyNet(torch_ref=th)
    models: List[Union[Net, SyNet]] = [sy_model, vanilla_model]
    for model in models:
        assert hasattr(model, "__call__")
        assert hasattr(model, "__init__")
        assert hasattr(model, "cpu")
        assert hasattr(model, "cuda")
        assert hasattr(model, "eval")
        assert hasattr(model, "forward")
        assert hasattr(model, "parameters")
        assert hasattr(model, "train")

        # check training is off
        assert model.training is False

        model.train()

        assert model.training is True

        model.eval()  # type: ignore

        assert model.training is False

    # assert vanilla_model.modules == sy_model._modules
    assert len(vanilla_model.modules) == len(sy_model.modules)
    assert len(sy_model.modules) == 6
    for syft_module, vanilla_module in zip(
        sy_model.modules.values(), vanilla_model.modules
    ):
        # we cant do module == module, but the str repr of the modules should be equal
        assert str(syft_module) == str(vanilla_module)
