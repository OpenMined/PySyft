import pytest

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import syft as sy


def test_send_train_config(hook):
    # To run a plan locally the local worker can't be a client worker,
    # since it needs to register objects
    hook.local_worker.is_client_worker = False

    # Send data and target to federated device
    data = th.tensor([[-1, 2.0]]).tag("data")
    target = th.tensor([[1.0]]).tag("target")
    federated_device = sy.VirtualWorker(hook, id="send_train_config", data=(data, target))

    # Loss function and model definition
    @sy.func2plan
    def loss_fn(real, pred):
        return ((real - pred) ** 2).mean()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        @sy.method2plan
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    # Create and send train config
    train_config = sy.TrainConfig(model=Net(), loss_plan=loss_fn)
    train_config.send(federated_device)

    # Get a pointer to federated device data
    pointer_to_data = federated_device.search("data")[0]
    pointer_to_target = federated_device.search("target")[0]

    # Run forward pass and calculate loss
    pointer_to_pred = train_config.forward_plan(pointer_to_data)
    pointer_to_loss = train_config.loss_plan(pointer_to_pred.wrap(), pointer_to_target)
    loss = pointer_to_loss.get()
    assert loss > 0


def test_run_train_config(hook):
    hook.local_worker.is_client_worker = False

    # Send data and target to federated device
    data = th.tensor([[-1, 2.0]])
    target = th.tensor([[1.0]])

    federated_device = sy.VirtualWorker(hook, id="run_train_config")
    ptr_data, ptr_target = data.send(federated_device), target.send(federated_device)
    dataset = sy.BaseDataset(ptr_data, ptr_target)
    federated_device.dataset = dataset

    # Loss function and model definition
    @sy.func2plan
    def loss_fn(real, pred):
        return ((real - pred) ** 2).mean()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        @sy.method2plan
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    # Force build
    loss_fn(th.tensor([1.0]), th.tensor([1.0]))

    # TODO: force forward build
    # x = th.tensor([1., 2])
    # print(model(x))
    # >       if input.dim() == 2 and bias is not None:
    # E   TypeError: __bool__ should return bool, returned Tensor

    # Create and send train config
    model = Net()
    train_config = sy.TrainConfig(model=model, loss_plan=loss_fn)
    train_config.send(federated_device)

    # TODO: uncomment this line
    # federated_device.fit()
