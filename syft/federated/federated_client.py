from typing import Union

import torch as th
from torch import nn

from syft.generic import ObjectStorage
from syft.federated.train_config import TrainConfig


class FederatedClient(ObjectStorage):
    """A Client able to execute federated learning in a defined dataset."""

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.optimizer = None
        self.train_config = None
        self.parameters = []

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

    def _build_optimizer(self, optimizer_name: str, lr: float) -> th.optim.Optimizer:
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
            self.optimizer = th.optim.SGD(self.parameters, lr=lr)
        else:
            raise ValueError("Unknown optimizer: {}".format(optimizer_name))
        return self.optimizer

    def fit_batch(self, *args, **kwargs):
        model_prediction = self.train_config.forward_plan(self._objects[b"#data"])
        model_prediction.owner.register_obj(model_prediction)
        loss = self.train_config.loss_plan(model_prediction, self._objects[b"#target"])
        return loss

    def fit(self, *args, **kwargs):
        if self.train_config is None:
            raise ValueError("TrainConfig not defined.")

        self._build_optimizer(self.train_config.optimizer, self.train_config.lr)
        self._fit()

    def _fit(self):
        # TODO: how to get the actual model?
        # self.model.train()
        # TODO: how to create/set a dataset?
        for data, target in self.dataset:
            self.optimizer.zero_grad()
            output = self.train_config.forward_plan(data)
            loss = self.train_config.loss_plan(output, target)
            loss.backward()
            self.optimizer.step()
        # TODO: check if multiple returns are supported.
        return self.model.get(), loss.get()
