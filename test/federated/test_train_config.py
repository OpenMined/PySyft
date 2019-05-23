import pytest

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import syft as sy


def test_train_config_with_jit(workers):
    alice = workers["alice"]
    me = workers["me"]

    data = th.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = th.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

    dataset = sy.BaseDataset(data, target)
    alice.add_dataset(dataset, key="vectors")

    @th.jit.script
    def loss_fn(real, pred):
        return ((real - pred) ** 2).mean()

    # Model
    class Net(th.jit.ScriptModule):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)

        @th.jit.script_method
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
    data = th.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = th.tensor([[1.0], [0.0], [1.0], [0.0]], requires_grad=True)

    print("Evaluation before training")
    pred = model(data)
    loss = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss))

    print("Evaluation after training:")
    pred = new_model(data)
    loss = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss))
