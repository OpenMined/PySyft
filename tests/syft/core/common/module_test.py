# stdlib
from typing import Any
from typing import List
from typing import Union

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
        def __init__(self) -> None:
            super(SyNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

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

    vanilla_model = Net()
    sy_model = SyNet()  # without the k it kant take over the world
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
    for _, module in sy_model.modules.items():
        # we cant do module == module, but the str repr of the modules should be equal
        assert str(module) == str(vanilla_model.modules.pop(0))
