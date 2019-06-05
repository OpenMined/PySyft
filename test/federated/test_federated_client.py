import pytest

import torch

import syft as sy
from syft import federated
from syft.frameworks.torch import pointers


def test_add_dataset():
    fed_client = federated.FederatedClient()

    dataset = "my_dataset"
    fed_client.add_dataset(dataset, "string_dataset")

    assert "string_dataset" in fed_client.datasets


def test_remove_dataset():
    fed_client = federated.FederatedClient()

    dataset = "my_dataset"
    key = "string_dataset"
    fed_client.add_dataset(dataset, key)

    assert key in fed_client.datasets

    fed_client.remove_dataset(key)

    assert key not in fed_client.datasets


def test_set_obj_train_config():
    fed_client = federated.FederatedClient()

    train_config = federated.TrainConfig(id=100, model=None, loss_fn=None)

    fed_client.set_obj(train_config)

    assert fed_client.train_config.id == train_config.id


def test_set_obj_other():
    fed_client = federated.FederatedClient()

    dummy_data = torch.tensor(42)
    dummy_data.id = 43

    fed_client.set_obj(dummy_data)

    assert len(fed_client._objects) == 1
    assert fed_client._objects[dummy_data.id] == dummy_data


@pytest.mark.skip(reason="bug in pytorch version 1.1.0, jit.trace returns raw C function")
def test_fit(hook):  # pragma: no cover
    fed_client = sy.VirtualWorker(hook=hook, id="test_fit_fc")

    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1.0], [0]], requires_grad=True)

    dataset = sy.BaseDataset(data, target)
    fed_client.add_dataset(dataset, key="data")

    @torch.jit.script
    def loss_fn(real, pred):
        return ((real.float() - pred.float()) ** 2).mean()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(2, 3)
            self.fc2 = torch.nn.Linear(3, 2)
            self.fc3 = torch.nn.Linear(2, 1)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_untraced = Net()
    model = torch.jit.trace(model_untraced, data)

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(batch_size=1, model=model, loss_fn=loss_fn)
    train_config.send(fed_client)

    print("test", fed_client.train_config._model_id)

    for epoch in range(5):
        loss = fed_client.fit(dataset_key="data")
        print("-" * 50)
        print("Iteration %s, loss: %s" % (epoch, loss))

    print("Evaluation after training:")
    new_model = train_config.model_ptr.get()
    pred = new_model.obj(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before
