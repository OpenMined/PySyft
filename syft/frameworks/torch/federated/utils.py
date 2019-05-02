import syft as sy
import torch
from typing import List
import logging

logger = logging.getLogger(__name__)


def extract_batches_per_worker(federated_train_loader: sy.FederatedDataLoader):
    """Extracts the batches from the federated_train_loader and stores them
       in a dictionary (keys = data.location)

       Args:
       federated_train_loader: the connection object we use to send responses
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
            dst_model (torch.nn.Module): the model to which the src_model will be added
            src_model (torch.nn.Module): the model to be added to dst_model
        Returns:
            torch.nn.Module: the resulting model of the addition

        """

    params1 = dst_model.named_parameters()
    params2 = src_model.named_parameters()
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.set_ = param1.data + dict_params2[name1].data
    return dst_model


def scale_model(model, scale):
    """Scale the parameters of a model.

    Args:
        model (torch.nn.Module): the models whose parameters will be scaled
        scale (float): the scaling factor
    Returns:
        torch.nn.Module: the module with scaled parameters

    """
    params = model.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        dict_params[name].data.set_ = dict_params[name].data * scale
    return model


def federated_avg(models: List[torch.nn.Module]):
    """Calculate the federated average of a list of models.

    Args:
        models (List[torch.nn.Module]): the models of which the federated average is calculated

    Returns:
        torch.nn.Module: the module with averaged parameters

    """
    nr_models = len(models)
    model_list = list(models.values())
    model = model_list[0]
    for i in range(1, nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model
