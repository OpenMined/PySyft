from typing import Union
import weakref

import torch
import syft as sy
from syft import workers
from syft.frameworks.torch import pointers


class TrainConfig:
    """TrainConfig abstraction.

    A wrapper object that contains all that is needed to run a training loop
    remotely on a federated learning setup.
    """

    def __init__(
        self,
        owner: workers.AbstractWorker = None,
        batch_size: int = 32,
        epochs: int = 1,
        optimizer: str = "sgd",
        lr: float = 0.1,
        id: Union[int, str] = None,
        loss_fn_id: int = None,
        model_id: int = None,
    ):
        """Initializer for TrainConfig.

        Args:
            batch_size: Batch size used for training.
            epochs: Epochs used for training.
            optimizer: A string indicating which optimizer should be used.
            lr: Learning rate.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the tensor.
            loss_fn_id: The id_at_location of (the ObjectWrapper of) a loss function which
                        shall be used to calculate the loss.
            model_id: id_at_location of a traced torch nn.Module instance (objectwrapper).
        """
        self.owner = owner if owner else sy.hook.local_worker
        self.id = id if id is not None else sy.ID_PROVIDER.pop()

        self.model_id = model_id
        self.loss_fn_id = loss_fn_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = lr
        self.location = None
        self.model_ptr = None
        self.loss_fn_ptr = None

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

    def send(
        self,
        location: workers.BaseWorker,
        traced_model: torch.jit.ScriptModule = None,
        traced_loss_fn: torch.jit.ScriptModule = None,
    ) -> weakref:
        """Gets the pointer to a new remote object.

        One of the most commonly used methods in PySyft, this method serializes
        the object upon which it is called (self), sends the object to a remote
        worker, creates a pointer to that worker, and then returns that pointer
        from this function.

        Args:
            location: The BaseWorker object which you want to send this object
                to. Note that this is never actually the BaseWorker but instead
                a class which instantiates the BaseWorker abstraction.
            traced_model: traced model to be sent jointly with the train configuration
            traced_loss_fn: traced loss function to be sent jointly with the train configuration.

        Returns:
            A weakref instance.
        """
        if traced_model:
            model_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=traced_model)
            self.model_ptr = self.owner.send(model_with_id, location)
            self.model_id = self.model_ptr.id_at_location
        if traced_loss_fn:
            loss_fn_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=traced_loss_fn)
            self.loss_fn_ptr = self.owner.send(loss_fn_with_id, location)
            self.loss_fn_id = self.loss_fn_ptr.id_at_location

        # Send train configuration itself
        ptr = self.owner.send(self, location)

        # TODO: why do we want to
        # we need to cache this weak reference to the pointer so that
        # if this method gets called multiple times we can simply re-use
        # the same pointer which was previously created
        self.ptr = weakref.ref(ptr)
        return self.ptr
