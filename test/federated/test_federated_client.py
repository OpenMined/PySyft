from syft import federated
import torch
import syft as sy
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

    train_config = federated.TrainConfig(id=10)

    fed_client.set_obj(train_config)

    assert fed_client.train_config.id == train_config.id


def test_set_obj_nn_param():
    fed_client = federated.FederatedClient()

    dummy_data = torch.tensor(42)
    param = torch.nn.Parameter(data=dummy_data, requires_grad=False)

    fed_client.set_obj(param)

    assert len(fed_client.parameters) == 1
    assert fed_client.parameters[0] == param


def test_set_obj_other():
    fed_client = federated.FederatedClient()

    dummy_data = torch.tensor(42)
    dummy_data.id = 43

    fed_client.set_obj(dummy_data)

    assert len(fed_client._objects) == 1
    assert fed_client._objects[dummy_data.id] == dummy_data


def test_fit():
    data = torch.tensor([[-1, 2.0], [0, 1.1], [-1, 2.1], [0, 1.2]], requires_grad=True)
    target = torch.tensor([[1], [0], [1], [0]])

    fed_client = federated.FederatedClient()
    dataset = sy.BaseDataset(data, target)
    fed_client.add_dataset(dataset, key="vectors")

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
    model_id = 0
    model_ow = pointers.ObjectWrapper(obj=model, id=model_id)
    loss_id = 1
    loss_ow = pointers.ObjectWrapper(obj=loss_fn, id=loss_id)

    print("Evaluation before training")
    pred = model(data)
    loss_before = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(batch_size=1, model_id=model_id, loss_fn_id=loss_id)

    fed_client.set_obj(model_ow)
    fed_client.set_obj(loss_ow)
    fed_client.set_obj(train_config)

    for epoch in range(5):
        loss = fed_client.fit(dataset="vectors")
        print("-" * 50)
        print("Iteration %s: alice's loss: %s" % (epoch, loss))

    print("Evaluation after training:")
    new_model = fed_client.get_obj(model_id)
    pred = new_model.obj(data)
    loss_after = loss_fn(real=target, pred=pred)
    print("Loss: {}".format(loss_after))

    assert loss_after < loss_before
