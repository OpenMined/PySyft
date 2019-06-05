import torch as th
from torch import nn
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import logging

from syft.generic import ObjectStorage
from syft.federated.train_config import TrainConfig

logger = logging.getLogger(__name__)


class FederatedClient(ObjectStorage):
    """A Client able to execute federated learning in a defined dataset."""

    def __init__(self, datasets=None, verbose=True):
        super().__init__()
        self.datasets = datasets if datasets is not None else dict()
        self.optimizer = None
        self.train_config = None
        self.parameters = []
        self.verbose = verbose

    def add_dataset(self, dataset, key: str):
        self.datasets[key] = dataset

    def remove_dataset(self, key: str):
        if key in self.datasets:
            del self.datasets[key]

    def set_obj(self, obj: object):
        """Registers objects checking if which objects it should cache.

        Args:
            obj: An object to be registered.
        """
        if isinstance(obj, TrainConfig):
            self.train_config = obj
            self.optimizer = None
        elif isinstance(obj, nn.parameter.Parameter):
            self.parameters.append(obj)
            super().set_obj(obj)
        else:
            super().set_obj(obj)

    def _build_optimizer(self, optimizer_name: str, model, lr: float) -> th.optim.Optimizer:
        """Build an optimizer if needed.

        Args:
            optimizer_name: A string indicating the optimizer name.
            lr: A float indicating the learning rate.
        Returns:
            A Torch Optimizer.
        """
        if self.optimizer is not None:
            return self.optimizer

        optimizer_name = optimizer_name.lower()
        if optimizer_name == "sgd":
            logger.debug("sgd parameters = %s, lr = %s", model.parameters(), lr)
            self.optimizer = th.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError("Unknown optimizer: {}".format(optimizer_name))
        return self.optimizer

    def fit(self, **kwargs):
        if self.train_config is None:
            raise ValueError("TrainConfig not defined.")
        if self.datasets is None:
            logger.error("No dataset available on worker")
            return None, -1
        try:
            key = kwargs["dataset"]
        except KeyError:
            logger.warning("Missing keyword argument 'dataset'.")
            if len(self.datasets) == 1:
                logger.warning("Only one dataset available, using this one.")
                key = list(self.datasets.keys())[0]
            else:
                logger.warning("Available datasets: %s", list(self.datasets.keys()))
                return None, -1

        model = self.get_obj(self.train_config._model_id).obj

        self._build_optimizer(self.train_config.optimizer, model, self.train_config.lr)
        return self._fit(model=model, key=key)

    def _create_batch_sampler(self, ds_key: str, shuffle: bool = False, drop_last: bool = True):
        data_range = range(len(self.datasets[ds_key]))
        if shuffle:
            sampler = RandomSampler(data_range)
        else:
            sampler = SequentialSampler(data_range)

        batch_sampler = BatchSampler(sampler, self.train_config.batch_size, drop_last)
        return batch_sampler

    def _fit(self, model, key):
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj
        model.train()
        batch_sampler = self._create_batch_sampler(key)
        loss = -1.0
        for data_indices in batch_sampler:
            data, target = self.datasets[key][data_indices]
            self.optimizer.zero_grad()
            output = model.forward(data)
            loss = loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
        return loss


#
