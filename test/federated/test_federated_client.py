import pytest

import torch

import syft as sy
from syft.federated.federated_client import FederatedClient
from syft.federated.train_config import TrainConfig
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.frameworks.torch.fl import utils

PRINT_IN_UNITTESTS = False

# To make execution deterministic to some extent
# For more information - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test_add_dataset():
    # Create a client to execute federated learning
    fed_client = FederatedClient()
    # Create a dataset
    dataset = "my_dataset"
    key = "string_dataset"
    # Add new dataset
    fed_client.add_dataset(dataset, key)

    assert "string_dataset" in fed_client.datasets


def test_add_dataset_with_duplicate_key():
    # Create a client to execute federated learning
    fed_client = FederatedClient()
    # Create a dataset
    dataset = "my_dataset"
    key = "string_dataset"
    # Add new dataset
    fed_client.add_dataset(dataset, key)

    assert "string_dataset" in fed_client.datasets
    # Raise an error if the key is already exists
    with pytest.raises(ValueError):
        fed_client.add_dataset(dataset, "string_dataset")


def test_remove_dataset():
    # Create a client to execute federated learning
    fed_client = FederatedClient()
    # Create a dataset
    dataset = "my_dataset"
    key = "string_dataset"
    # Add new dataset
    fed_client.add_dataset(dataset, key)

    assert key in fed_client.datasets
    # Remove new dataset
    fed_client.remove_dataset(key)

    assert key not in fed_client.datasets


def test_set_obj_train_config():
    fed_client = FederatedClient()
    fed_client.optimizer = True

    train_config = TrainConfig(id=100, model=None, loss_fn=None)

    fed_client.set_obj(train_config)

    assert fed_client.train_config.id == train_config.id
    assert fed_client.optimizer is None


def test_set_obj_other():
    fed_client = FederatedClient()

    dummy_data = torch.tensor(42)
    dummy_data.id = 43

    fed_client.set_obj(dummy_data)

    assert len(fed_client.object_store._objects) == 1
    assert fed_client.object_store.get_obj(dummy_data.id) == dummy_data


def evaluate_model(fed_client, model_id, loss_fn, data, target):  # pragma: no cover
    new_model = fed_client.get_obj(model_id)
    pred = new_model.obj(data)
    loss_after = loss_fn(target=target, pred=pred)
    return loss_after


def train_model(
    fed_client, fit_dataset_key, available_dataset_key, nr_rounds, device
):  # pragma: no cover
    loss = None
    for curr_round in range(nr_rounds):
        if fit_dataset_key == available_dataset_key:
            loss = fed_client.fit(dataset_key=fit_dataset_key, device=device)
        else:
            with pytest.raises(ValueError):
                loss = fed_client.fit(dataset_key=fit_dataset_key, device=device)
        if PRINT_IN_UNITTESTS and curr_round % 2 == 0:  # pragma: no cover
            print("-" * 50)
            print(f"Iteration {curr_round}: alice's loss: {loss}")


@pytest.mark.parametrize(
    "fit_dataset_key, epochs, device",
    [
        ("gaussian_mixture", 1, "cpu"),
        ("gaussian_mixture", 10, "cpu"),
        ("another_dataset", 1, "cpu"),
        ("gaussian_mixture", 10, "cuda"),
    ],
)
def test_fit(fit_dataset_key, epochs, device):

    if device == "cuda" and not torch.cuda.is_available():
        return

    data, target = utils.create_gaussian_mixture_toy_data(nr_samples=100)

    fed_client = FederatedClient()
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

    data_device = data.to(torch.device(device))
    target_device = target.to(torch.device(device))
    model_untraced = Net().to(torch.device(device))
    model = torch.jit.trace(model_untraced, data_device)
    model_id = 0
    model_ow = ObjectWrapper(obj=model, id=model_id)
    loss_id = 1
    loss_ow = ObjectWrapper(obj=loss_fn, id=loss_id)
    pred = model(data_device)
    loss_before = loss_fn(target=target_device, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print(f"Loss before training: {loss_before}")

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

    train_model(
        fed_client, fit_dataset_key, available_dataset_key=dataset_key, nr_rounds=3, device=device
    )

    if dataset_key == fit_dataset_key:
        loss_after = evaluate_model(fed_client, model_id, loss_fn, data_device, target_device)
        if PRINT_IN_UNITTESTS:  # pragma: no cover
            print(f"Loss after training: {loss_after}")

        if loss_after >= loss_before:  # pragma: no cover
            if PRINT_IN_UNITTESTS:
                print(f"Loss not reduced, train more: {loss_after}")

            train_model(
                fed_client,
                fit_dataset_key,
                available_dataset_key=dataset_key,
                nr_rounds=10,
                device=device,
            )
            loss_after = evaluate_model(fed_client, model_id, loss_fn, data, target)

        assert loss_after <= loss_before


def test_evaluate():  # pragma: no cover
    data, target = utils.iris_data_partial()

    fed_client = FederatedClient()
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
    model_ow = ObjectWrapper(obj=model, id=model_id)
    loss_id = 1
    loss_ow = ObjectWrapper(obj=loss_fn, id=loss_id)
    pred = model(data)
    loss_before = loss_fn(target=target, pred=pred)
    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print(f"Loss before training: {loss_before}")

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
        print(f"Evaluation result before training: {result}")

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
    train_model(
        fed_client, dataset_key, available_dataset_key=dataset_key, nr_rounds=50, device="cpu"
    )

    result = fed_client.evaluate(
        dataset_key=dataset_key, return_histograms=True, nr_bins=3, return_loss=True
    )

    test_loss_after = result["loss"]
    correct_after = result["nr_correct_predictions"]
    len_dataset = result["nr_predictions"]
    hist_pred_after = result["histogram_predictions"]
    hist_target = result["histogram_target"]

    if PRINT_IN_UNITTESTS:  # pragma: no cover
        print(f"Evaluation result: {result}")

    assert len_dataset == 30
    assert (hist_target == [10, 10, 10]).all()
    assert correct_after > correct_before
    assert torch.norm(torch.tensor(hist_target - hist_pred_after)) < torch.norm(
        torch.tensor(hist_target - hist_pred_before)
    )
