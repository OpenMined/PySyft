import torch

import syft as sy
from syft import federated
from syft.frameworks.torch import pointers
from syft.frameworks.torch.federated import utils

PRINT_IN_UNITTESTS = False


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
    fed_client.optimizer = True

    train_config = federated.TrainConfig(id=100, model=None, loss_fn=None)

    fed_client.set_obj(train_config)

    assert fed_client.train_config.id == train_config.id
    assert fed_client.optimizer is None


def test_set_obj_other():
    fed_client = federated.FederatedClient()

    dummy_data = torch.tensor(42)
    dummy_data.id = 43

    fed_client.set_obj(dummy_data)

    assert len(fed_client._objects) == 1
    assert fed_client._objects[dummy_data.id] == dummy_data


def test_fit():
    data, target = utils.create_gaussian_mixture_toy_data(nr_samples=100)

    fed_client = federated.FederatedClient()
    dataset = sy.BaseDataset(data, target)
    dataset_key = "gaussian_mixture"
    fed_client.add_dataset(dataset, key=dataset_key)

    def loss_fn(target, pred):
        return torch.nn.functional.cross_entropy(input=pred, target=target)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(2, 3)
            self.fc2 = torch.nn.Linear(3, 2)

            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.xavier_normal_(self.fc2.weight)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            return x

    model_untraced = Net()
    model = torch.jit.trace(model_untraced, data)
    model_id = 0
    model_ow = pointers.ObjectWrapper(obj=model, id=model_id)
    loss_id = 1
    loss_ow = pointers.ObjectWrapper(obj=loss_fn, id=loss_id)
    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Loss before training: {}".format(loss_before))

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=8,
        model=None,
        loss_fn=None,
        model_id=model_id,
        loss_fn_id=loss_id,
        lr=0.05,
        weight_decay=0.01,
    )

    fed_client.set_obj(model_ow)
    fed_client.set_obj(loss_ow)
    fed_client.set_obj(train_config)
    fed_client.optimizer = None

    for curr_round in range(12):
        loss = fed_client.fit(dataset_key=dataset_key)
        if PRINT_IN_UNITTESTS and curr_round % 4 == 0:  # pragma: no cover
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (curr_round, loss))

    new_model = fed_client.get_obj(model_id)
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover:
        print("Loss after training: {}".format(loss_after))

    assert loss_after < loss_before


def create_xor_data(nr_samples):  # pragma: no cover
    with torch.no_grad():
        data = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], requires_grad=True)
        target = torch.tensor([1, 1, 0, 0], requires_grad=False)

        data_len = int(nr_samples / 4 + 1) * 4
        X = torch.zeros(data_len, 2, requires_grad=True)
        Y = torch.zeros(data_len, requires_grad=False)

        for i in range(int(data_len / 4)):
            X[i * 4 : (i + 1) * 4, :] = data
            Y[i * 4 : (i + 1) * 4] = target

    return X, Y.long()
