import pytest

import torch

import syft as sy
from syft import federated
from syft.frameworks.torch import pointers
from syft.frameworks.torch.federated import utils

PRINT_IN_UNITTESTS = False

# To make execution deterministic to some extent
# for more information - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test_add_dataset():
    fed_client = federated.FederatedClient()

    dataset = "my_dataset"
    fed_client.add_dataset(dataset, "string_dataset")

    assert "string_dataset" in fed_client.datasets


def test_add_dataset_with_duplicate_key():
    fed_client = federated.FederatedClient()

    dataset = "my_dataset"
    fed_client.add_dataset(dataset, "string_dataset")

    assert "string_dataset" in fed_client.datasets

    with pytest.raises(ValueError):
        fed_client.add_dataset(dataset, "string_dataset")


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


def evaluate_model(fed_client, model_id, loss_fn, data, target):  # pragma: no cover
    new_model = fed_client.get_obj(model_id)
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)
    return loss_after


def train_model(fed_client, fit_dataset_key, available_dataset_key, nr_rounds):  # pragma: no cover
    loss = None
    for curr_round in range(nr_rounds):
        if fit_dataset_key == available_dataset_key:
            loss = fed_client.fit(dataset_key=fit_dataset_key)
        else:
            with pytest.raises(ValueError):
                loss = fed_client.fit(dataset_key=fit_dataset_key)
        if PRINT_IN_UNITTESTS and curr_round % 2 == 0:  # pragma: no cover
            print("-" * 50)
            print("Iteration %s: alice's loss: %s" % (curr_round, loss))


@pytest.mark.parametrize(
    "fit_dataset_key, epochs",
    [("gaussian_mixture", 1), ("gaussian_mixture", 10), ("another_dataset", 1)],
)
def test_fit(fit_dataset_key, epochs):
    data, target = utils.create_gaussian_mixture_toy_data(nr_samples=100)

    fed_client = federated.FederatedClient()
    dataset = sy.BaseDataset(data, target)
    dataset_key = "gaussian_mixture"
    fed_client.add_dataset(dataset, key=dataset_key)

    def loss_fn(pred, target):
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
        optimizer_args={"lr": 0.05, "weight_decay": 0.01},
        epochs=epochs,
    )

    fed_client.set_obj(model_ow)
    fed_client.set_obj(loss_ow)
    fed_client.set_obj(train_config)
    fed_client.optimizer = None

    train_model(fed_client, fit_dataset_key, available_dataset_key=dataset_key, nr_rounds=3)

    if dataset_key == fit_dataset_key:
        loss_after = evaluate_model(fed_client, model_id, loss_fn, data, target)
        if PRINT_IN_UNITTESTS:  # pragma: no cover
            print("Loss after training: {}".format(loss_after))

        if loss_after >= loss_before:  # pragma: no cover
            if PRINT_IN_UNITTESTS:
                print("Loss not reduced, train more: {}".format(loss_after))

            train_model(
                fed_client, fit_dataset_key, available_dataset_key=dataset_key, nr_rounds=10
            )
            loss_after = evaluate_model(fed_client, model_id, loss_fn, data, target)

        assert loss_after < loss_before


@pytest.mark.skipif(
    torch.__version__ >= "1.1",
    reason="bug in pytorch version 1.1.0, jit.trace returns raw C function",
)
def test_evaluate():  # pragma: no cover
    data, target = utils.iris_data_partial()

    fed_client = federated.FederatedClient()
    dataset = sy.BaseDataset(data, target)
    dataset_key = "iris"
    fed_client.add_dataset(dataset, key=dataset_key)

    def loss_fn(pred, target):
        return torch.nn.functional.cross_entropy(input=pred, target=target)

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(4, 3)

        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            return x

    model_untraced = Net()

    with torch.no_grad():
        model_untraced.fc1.weight.set_(
            torch.tensor(
                [
                    [0.0160, 1.3753, -0.1202, -0.9129],
                    [0.1539, 0.3092, 0.0749, 0.2142],
                    [0.0984, 0.6248, 0.0274, 0.1735],
                ]
            )
        )
        model_untraced.fc1.bias.set_(torch.tensor([0.3477, 0.2970, -0.0799]))

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
        optimizer_args=None,
        epochs=1,
    )

    fed_client.set_obj(model_ow)
    fed_client.set_obj(loss_ow)
    fed_client.set_obj(train_config)
    fed_client.optimizer = None

    result = fed_client.evaluate(
        dataset_key=dataset_key, return_histograms=True, nr_bins=3, return_loss=True
    )

    test_loss_before = result["loss"]
    correct_before = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred_before = result["histogram_predictions"]
    hist_target = result["histogram_target"]

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Evaluation result before training: {}".format(result))

    assert len_dataset == 30
    assert (hist_target == [10, 10, 10]).all()

    train_config = sy.TrainConfig(
        batch_size=8,
        model=None,
        loss_fn=None,
        model_id=model_id,
        loss_fn_id=loss_id,
        optimizer="SGD",
        optimizer_args={"lr": 0.01},
        shuffle=True,
        epochs=2,
    )
    fed_client.set_obj(train_config)
    train_model(fed_client, dataset_key, available_dataset_key=dataset_key, nr_rounds=50)

    result = fed_client.evaluate(
        dataset_key=dataset_key, return_histograms=True, nr_bins=3, return_loss=True
    )

    test_loss_after = result["loss"]
    correct_after = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred_after = result["histogram_predictions"]
    hist_target = result["histogram_target"]

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print("Evaluation result: {}".format(result))

    assert len_dataset == 30
    assert (hist_target == [10, 10, 10]).all()
    assert correct_after > correct_before
    assert torch.norm(torch.tensor(hist_target - hist_pred_after)) < torch.norm(
        torch.tensor(hist_target - hist_pred_before)
    )
