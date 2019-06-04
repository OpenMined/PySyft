import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy

from syft.frameworks.torch import pointers


@pytest.mark.skip(reason="fails currently as it needs functions as torch.nn.linear to be unhooked.")
def test_train_config_with_jit_script_module(hook, workers):
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @hook.torch.jit.script
    def loss_fn(real, pred):
        return ((real - pred) ** 2).mean()

    # Model
    class Net(torch.jit.ScriptModule):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        @torch.jit.script_method
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = Net()
    model.id = sy.ID_PROVIDER.pop()

    loss_fn.id = sy.ID_PROVIDER.pop()

    model_ptr = me.send(model, alice)
    loss_fn_ptr = me.send(loss_fn, alice)

    # Create and send train config
    train_config = sy.TrainConfig(
        model_id=model_ptr.id_at_location, loss_plan_id=loss_fn_ptr.id_at_location, batch_size=2
    )
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key="vectors")
        print("-" * 50)
        print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print(alice)
    new_model = model_ptr.get()
    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    print("Evaluation after training:")
    pred = new_model(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before


def test_train_config_with_jit_trace(hook, workers):
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @hook.torch.jit.script
    def loss_fn(real, pred):
        return ((real - pred) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()

    model = torch.jit.trace(model_untraced, data)
    model_with_id = pointers.ObjectWrapper(model, sy.ID_PROVIDER.pop())

    loss_fn_with_id = pointers.ObjectWrapper(loss_fn, sy.ID_PROVIDER.pop())

    model_ptr = me.send(model_with_id, alice)
    loss_fn_ptr = me.send(loss_fn_with_id, alice)

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(
        model_id=model_ptr.id_at_location, loss_plan_id=loss_fn_ptr.id_at_location, batch_size=2
    )
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key="vectors")
        print("-" * 50)
        print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print("Evaluation after training:")
    new_model = model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before
