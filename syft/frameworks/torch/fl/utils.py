import syft as sy
import torch
from typing import Dict
from typing import Any
import logging

logger = logging.getLogger(__name__)


def extract_batches_per_worker(federated_train_loader: sy.FederatedDataLoader):
    """Extracts the batches from the federated_train_loader and stores them
       in a dictionary (keys = data.location).

    Args:
        federated_train_loader: the connection object we use to send responses.
                back to the client.
    """
    logging_interval = 100
    batches = {}
    for worker_id in federated_train_loader.workers:
        worker = federated_train_loader.federated_dataset.datasets[worker_id].location
        batches[worker] = []

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        if batch_idx % logging_interval == 0:
            logger.debug("Extracted %s batches from federated_train_loader", batch_idx)
        batches[data.location].append((data, target))

    return batches


def add_model(dst_model, src_model):
    """Add the parameters of two models.

    Args:
        dst_model (torch.nn.Module): the model to which the src_model will be added.
        src_model (torch.nn.Module): the model to be added to dst_model.
    Returns:
        torch.nn.Module: the resulting model of the addition.

    """

    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.

    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def federated_avg(models: Dict[Any, torch.nn.Module]) -> torch.nn.Module:
    """Calculate the federated average of a dictionary containing models.
       The models are extracted from the dictionary
       via the models.values() command.

    Args:
        models (Dict[Any, torch.nn.Module]): a dictionary of models
        for which the federated average is calculated.

    Returns:
        torch.nn.Module: the module with averaged parameters.
    """
    nr_models = len(models)
    model_list = list(models.values())
    model = type(model_list[0])()

    for i in range(nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model


def accuracy(pred_softmax, target):
    """Calculate the accuray of a given prediction.

    This functions assumes pred_softmax to be converted into the final prediction by
    taking the argmax.

    Args:
        pred_softmax: array type(float), providing nr_classes values per element in target.
        target: array type(int), correct classes, taking values in range [0, nr_classes).

    Returns:
        accuracy: float, fraction of correct predictions.

    """
    nr_elems = len(target)
    pred = pred_softmax.argmax(dim=1)
    return (pred.float() == target.view(pred.shape).float()).sum().numpy() / float(nr_elems)


def create_gaussian_mixture_toy_data(nr_samples: int):  # pragma: no cover
    """Create a simple toy data for binary classification

    The data is drawn from two normal distributions
    target = 1: mu = 2, sigma = 1
    target = 0: mu = 0, sigma = 1
    The dataset is balanced with an equal number of positive and negative samples

    Args:
        nr_samples: number of samples to generate

    Returns:
        data, targets


    """
    sample_dim = 2
    one_half = int(nr_samples / 2)
    x1 = torch.randn(one_half, sample_dim, requires_grad=True) - 5
    x2 = torch.randn(one_half, sample_dim, requires_grad=True) + 5
    x = torch.cat([x1, x2], dim=0)
    y1 = torch.zeros(one_half, requires_grad=False).long()
    y2 = torch.ones(one_half, requires_grad=False).long()
    y = torch.cat([y1, y2], dim=0)
    return x, y


def iris_data_partial():
    """

    Returns: 30 samples from the iris data set: https://archive.ics.uci.edu/ml/datasets/iris

    """
    data = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
    ]

    targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    data += [
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6],
        [4.9, 2.4, 3.3, 1.0],
        [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4],
    ]

    targets += [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    data += [
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
        [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7],
        [7.3, 2.9, 6.3, 1.8],
        [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5],
    ]

    targets += [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    return torch.tensor(data), torch.tensor(targets)
