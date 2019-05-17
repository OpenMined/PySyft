import random
from typing import List
from typing import Union
import weakref

import torch
import torch.nn as nn

import syft
import syft as sy
from syft.federated import Plan
from syft.frameworks.torch.tensors.interpreters import AbstractTensor
from syft.frameworks.torch.tensors.interpreters import PointerTensor
from syft.workers import AbstractWorker

from syft.exceptions import PureTorchTensorFoundError


class TrainConfig:
    """TrainConfig abstraction.

    A wrapper object that contains all that is needed to run a training loop
    remotely on a federated learning setup.
    """

    def __init__(
        self,
        loss_plan: Plan,
        model: nn.Module = None,
        forward_plan: Plan = None,
        batch_size: int = 32,
        epochs: int = 1,
        optimizer: str = "sgd",
        lr: float = 0.1,
        owner: AbstractWorker = None,
        id: Union[int, str] = None,
    ):
        """Initializer for TrainConfig.

        Args:
            loss_plan: A plan containing how the loss should be calculated.
            model: A torch nn.Module instance.
            forward_plan: A plan containg how to perform the model forward computation.
                If the model is provided this argument is ignored and `model.forward`
                is used instead.
            batch_size: Batch size used for training.
            epochs: Epochs used for training.
            optimizer: A string indicating which optimizer should be used.
            lr: Learning rate.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the tensor.
        """
        self.owner = owner if owner else sy.hook.local_worker
        self.id = id if id is not None else sy.ID_PROVIDER.pop()

        self.model = model
        self.forward_plan = model.forward if model else forward_plan
        self.loss_plan = loss_plan
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = lr
        self.location = None

    def __str__(self) -> str:
        """Returns the string representation of a TrainConfig."""
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " owner:" + str(self.owner.id)

        if self.location:
            out += " location:" + str(self.location.id)

        out += " epochs: " + str(self.epochs)
        out += " batch_size: " + str(self.batch_size)
        out += " lr: " + str(self.lr)

        out += ">"
        return out

    def send(self, location: "syft.workers.BaseWorker") -> weakref:
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.

        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.

        Returns:
            A weakref instance.
        """
        # Send Model
        self.model.send(location)

        # Send plans and cache them so they can be reused
        # when this trainConfig instance is sent to location
        self.forward_plan = self.model.forward._send(location)
        self.loss_plan = self.loss_plan._send(location)

        # Send train configuration itself
        ptr = self.owner.send(self, location)

        # we need to cache this weak reference to the pointer so that
        # if this method gets called multiple times we can simply re-use
        # the same pointer which was previously created
        self.ptr = weakref.ref(ptr)
        return self.ptr
