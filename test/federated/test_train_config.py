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
    federated_device = sy.VirtualWorker(hook, id="federated_device", data=(data, target))

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
    pointer_to_loss.get()
    assert True
