import pytest

from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy

from syft.frameworks.torch import pointers


@pytest.mark.skip(reason="fails currently as it needs functions as torch.nn.linear to be unhooked.")
def test_train_config_with_jit_script_module(hook, workers):  # pragma: no cover
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]], requires_grad=True)

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @hook.torch.jit.script
    def loss_fn(real, pred):
        return ((real.float() - pred.float()) ** 2).mean()

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
        model_id=model_ptr.id_at_location, loss_fn_id=loss_fn_ptr.id_at_location, batch_size=2
    )
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset="vectors")
        print("-" * 50)
        print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print(alice)
    new_model = model_ptr.get()
    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    print("Evaluation after training:")
    pred = new_model(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before


@pytest.mark.skip(reason="bug in pytorch version 1.1.0, jit.trace returns raw C function")
def test_train_config_with_jit_trace(hook, workers):  # pragma: no cover
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @hook.torch.jit.script
    def loss_fn(real, pred):
        return ((real.float() - pred.float()) ** 2).mean()

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

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(
        model_id=model_ptr.id_at_location, loss_fn_id=loss_fn_ptr.id_at_location, batch_size=2
    )
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset="vectors")
        print("-" * 50)
        print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print("Evaluation after training:")
    new_model = model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before


@pytest.mark.skip(reason="bug in pytorch version 1.1.0, jit.trace returns raw C function")
def test_train_config_send_with_traced_fns(hook, workers):  # pragma: no cover
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @hook.torch.jit.script
    def loss_fn(real, pred):
        return ((real.float() - pred.float()) ** 2).mean()

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

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    # Create and send train config
    traced_model = torch.jit.trace(model, data)
    train_config = sy.TrainConfig(batch_size=2)
    train_config.send(alice, traced_loss_fn=loss_fn, traced_model=traced_model)

    for epoch in range(5):
        loss = alice.fit(dataset="vectors")
        print("-" * 50)
        print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print("Evaluation after training:")
    new_model = train_config.model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before


def test___str__():
    train_config = sy.TrainConfig(batch_size=2, id=99887766)

    train_config_str = str(train_config)
    str_expected = "<TrainConfig id:99887766 owner:me epochs: 1 batch_size: 2 lr: 0.1>"

    assert str_expected == train_config_str


def test_send(workers):
    id = 10

    train_config = sy.TrainConfig(batch_size=2, id=id)
    alice = workers["alice"]
    train_config.send(alice)

    assert alice.train_config.id == train_config.id
    assert alice.train_config.model_id == train_config.model_id
    assert alice.train_config.loss_fn_id == train_config.loss_fn_id
    assert alice.train_config.batch_size == train_config.batch_size
    assert alice.train_config.epochs == train_config.epochs
    assert alice.train_config.optimizer == train_config.optimizer
    assert alice.train_config.lr == train_config.lr
    assert alice.train_config.location == train_config.location


def test_send_model_and_loss_fn(workers):
    id = 10

    model = 4
    loss_fn = 5

    train_config = sy.TrainConfig(batch_size=2, id=id)
    alice = workers["alice"]

    orig_func = sy.ID_PROVIDER.pop
    model_id = 44
    model_id_at_location = 44000
    loss_fn_id = 55
    loss_fn_id_at_location = 55000
    sy.ID_PROVIDER.pop = mock.Mock(
        side_effect=[model_id, model_id_at_location, loss_fn_id, loss_fn_id_at_location]
    )

    train_config.send(alice, traced_model=model, traced_loss_fn=loss_fn)

    assert alice.train_config.id == train_config.id
    assert alice.train_config.model_id == train_config.model_id
    assert alice.train_config.loss_fn_id == train_config.loss_fn_id
    assert alice.train_config.batch_size == train_config.batch_size
    assert alice.train_config.epochs == train_config.epochs
    assert alice.train_config.optimizer == train_config.optimizer
    assert alice.train_config.lr == train_config.lr
    assert alice.train_config.location == train_config.location
    assert alice.train_config.model_id == model_id
    assert alice.train_config.loss_fn_id == loss_fn_id
    assert alice.get_obj(model_id).obj == model
    assert alice.get_obj(loss_fn_id).obj == loss_fn

    sy.ID_PROVIDER.pop = orig_func
