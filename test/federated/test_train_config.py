import pytest

from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy

import time
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker
from syft.frameworks.torch.federated import utils

PRINT_IN_UNITTESTS = False


@pytest.mark.skip(reason="fails currently as it needs functions as torch.nn.linear to be unhooked.")
def test_train_config_with_jit_script_module(hook, workers):  # pragma: no cover
    alice = workers["alice"]
    me = workers["me"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

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
    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2)
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset="vectors")
        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print(alice)
    new_model = model_ptr.get()
    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)

    pred = new_model(data)
    loss_after = loss_fn(real=target, pred=pred)

    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))

    assert loss_after < loss_before


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_train_config_with_jit_trace(hook, workers):  # pragma: no cover
    alice = workers["alice"]

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="gaussian_mixture")

    @hook.torch.jit.script
    def loss_fn(pred, target):
        return ((target.float() - pred.float()) ** 2).mean()

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
    loss_before = loss_fn(target=target, pred=pred)
    print("Loss: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2)
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key="gaussian_mixture")
        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)

    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))

    assert loss_after < loss_before


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_train_config_with_jit_trace_send_twice_with_fit(hook, workers):  # pragma: no cover
    alice = workers["alice"]
    model, loss_fn, data, target, loss_before, dataset_key = prepare_training(hook, alice)

    # Create and send train config
    train_config_0 = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2)
    train_config_0.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("TrainConfig 0, iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config_0.model_ptr.get()
    pred = new_model.obj(data)
    loss_after_0 = loss_fn(pred=pred, target=target)

    assert loss_after_0 < loss_before

    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2)
    train_config.send(alice)

    for epoch in range(5):
        loss = alice.fit(dataset_key=dataset_key)

        if PRINT_IN_UNITTESTS:  # pragma: no cover:
            print("-" * 50)
            print("TrainConfig 1, iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(pred=pred, target=target)
    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss after training with TrainConfig 0: {}".format(loss_after_0))
        print("Loss after training with TrainConfig 1:   {}".format(loss_after))

    assert loss_after < loss_before


def prepare_training(hook, alice):  # pragma: no cover

    data, target = utils.create_gaussian_mixture_toy_data(nr_samples=100)
    dataset_key = "gaussian_mixture"

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key=dataset_key)

    @hook.torch.jit.script
    def loss_fn(pred, target):
        return ((target.float() - pred.float()) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()

    model = torch.jit.trace(model_untraced, data)

    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)
    return model, loss_fn, data, target, loss_before, dataset_key


def test___str__():
    train_config = sy.TrainConfig(batch_size=2, id=99887766, model=None, loss_fn=None)

    train_config_str = str(train_config)
    str_expected = (
        "<TrainConfig id:99887766 owner:me epochs: 1 batch_size: 2 optimizer_args: {'lr': 0.1}>"
    )

    assert str_expected == train_config_str


def test_send(workers):
    alice = workers["alice"]

    train_config = sy.TrainConfig(batch_size=2, id="id", model=None, loss_fn=None)
    train_config.send(alice)

    assert alice.train_config.id == train_config.id
    assert alice.train_config._model_id == train_config._model_id
    assert alice.train_config._loss_fn_id == train_config._loss_fn_id
    assert alice.train_config.batch_size == train_config.batch_size
    assert alice.train_config.epochs == train_config.epochs
    assert alice.train_config.optimizer == train_config.optimizer
    assert alice.train_config.optimizer_args == train_config.optimizer_args
    assert alice.train_config.location == train_config.location


def test_send_model_and_loss_fn(workers):
    train_config = sy.TrainConfig(
        batch_size=2, id="send_model_and_loss_fn_tc", model=None, loss_fn=None
    )
    alice = workers["alice"]

    orig_func = sy.ID_PROVIDER.pop
    model_id = 44
    model_id_at_location = 44000
    loss_fn_id = 55
    loss_fn_id_at_location = 55000
    sy.ID_PROVIDER.pop = mock.Mock(
        side_effect=[model_id, model_id_at_location, loss_fn_id, loss_fn_id_at_location]
    )

    train_config.send(alice)

    assert alice.train_config.id == train_config.id
    assert alice.train_config._model_id == train_config._model_id
    assert alice.train_config._loss_fn_id == train_config._loss_fn_id
    assert alice.train_config.batch_size == train_config.batch_size
    assert alice.train_config.epochs == train_config.epochs
    assert alice.train_config.optimizer == train_config.optimizer
    assert alice.train_config.optimizer_args == train_config.optimizer_args
    assert alice.train_config.location == train_config.location
    assert alice.train_config._model_id == model_id
    assert alice.train_config._loss_fn_id == loss_fn_id

    sy.ID_PROVIDER.pop = orig_func


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
@pytest.mark.asyncio
async def test_train_config_with_jit_trace_async(hook, start_proc):  # pragma: no cover
    kwargs = {"id": "async_fit", "host": "localhost", "port": 8777, "hook": hook}
    # data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
    # target = torch.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=False)
    # dataset_key = "xor"
    data, target = utils.create_gaussian_mixture_toy_data(100)
    dataset_key = "gaussian_mixture"

    mock_data = torch.zeros(1, 2)

    # TODO check reason for error (RuntimeError: This event loop is already running) when starting websocket server from pytest-asyncio environment
    # dataset = sy.BaseDataset(data, target)

    # process_remote_worker = start_proc(WebsocketServerWorker, dataset=(dataset, dataset_key), **kwargs)

    # time.sleep(0.1)

    local_worker = WebsocketClientWorker(**kwargs)

    @hook.torch.jit.script
    def loss_fn(pred, target):
        return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

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

    model = torch.jit.trace(model_untraced, mock_data)

    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)

    # Create and send train config
    train_config = sy.TrainConfig(
        model=model, loss_fn=loss_fn, batch_size=2, optimizer="SGD", optimizer_args={"lr": 0.1}
    )
    train_config.send(local_worker)

    for epoch in range(5):
        loss = await local_worker.async_fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS:  # pragma: no cover
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()

    assert not (model.fc1._parameters["weight"] == new_model.obj.fc1._parameters["weight"]).all()
    assert not (model.fc2._parameters["weight"] == new_model.obj.fc2._parameters["weight"]).all()
    assert not (model.fc3._parameters["weight"] == new_model.obj.fc3._parameters["weight"]).all()
    assert not (model.fc1._parameters["bias"] == new_model.obj.fc1._parameters["bias"]).all()
    assert not (model.fc2._parameters["bias"] == new_model.obj.fc2._parameters["bias"]).all()
    assert not (model.fc3._parameters["bias"] == new_model.obj.fc3._parameters["bias"]).all()

    new_model.obj.eval()
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))

    local_worker.ws.shutdown()
    # process_remote_worker.terminate()

    assert loss_after < loss_before


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_train_config_with_jit_trace_sync(hook, start_proc):  # pragma: no cover
    kwargs = {"id": "sync_fit", "host": "localhost", "port": 9000, "hook": hook}

    data, target = utils.create_gaussian_mixture_toy_data(100)

    dataset = sy.BaseDataset(data, target)

    dataset_key = "gaussian_mixture"
    process_remote_worker = start_proc(
        WebsocketServerWorker, dataset=(dataset, dataset_key), **kwargs
    )

    time.sleep(0.1)

    local_worker = WebsocketClientWorker(**kwargs)

    @hook.torch.jit.script
    def loss_fn(pred, target):
        return ((target.view(pred.shape).float() - pred.float()) ** 2).mean()

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

    pred = model(data)
    loss_before = loss_fn(pred=pred, target=target)

    # Create and send train config
    train_config = sy.TrainConfig(model=model, loss_fn=loss_fn, batch_size=2, epochs=1)
    train_config.send(local_worker)

    for epoch in range(5):
        loss = local_worker.fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS:  # pragma: no cover
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (epoch, loss))

    new_model = train_config.model_ptr.get()

    # assert that the new model has updated (modified) parameters
    assert not (model.fc1._parameters["weight"] == new_model.obj.fc1._parameters["weight"]).all()
    assert not (model.fc2._parameters["weight"] == new_model.obj.fc2._parameters["weight"]).all()
    assert not (model.fc3._parameters["weight"] == new_model.obj.fc3._parameters["weight"]).all()
    assert not (model.fc1._parameters["bias"] == new_model.obj.fc1._parameters["bias"]).all()
    assert not (model.fc2._parameters["bias"] == new_model.obj.fc2._parameters["bias"]).all()
    assert not (model.fc3._parameters["bias"] == new_model.obj.fc3._parameters["bias"]).all()

    new_model.obj.eval()
    pred = new_model.obj(data)
    loss_after = loss_fn(pred=pred, target=target)

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Loss before training: {}".format(loss_before))
        print("Loss after training: {}".format(loss_after))

    local_worker.ws.shutdown()
    del local_worker

    process_remote_worker.terminate()

    assert loss_after < loss_before
