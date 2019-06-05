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
        model: torch.jit.ScriptModule,
        loss_fn: torch.jit.ScriptModule,
        owner: workers.AbstractWorker = None,
        batch_size: int = 32,
        epochs: int = 1,
        optimizer: str = "sgd",
        lr: float = 0.1,
        id: Union[int, str] = None,
        max_nr_batches: int = -1,
        shuffle: bool = True,
        loss_fn_id: int = None,
        model_id: int = None,
    ):
        """Initializer for TrainConfig.

        Args:
            model: A traced torch nn.Module instance.
            loss_fn: A jit function representing a loss function which
                shall be used to calculate the loss.
            batch_size: Batch size used for training.
            epochs: Epochs used for training.
            optimizer: A string indicating which optimizer should be used.
            lr: Learning rate.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the tensor.
            max_nr_batches: Maximum number of training steps that will be performed. For large datasets
                            this can be used to run for less than the number of epochs provided.
            shuffle: boolean, whether to access the dataset randomly (shuffle) or sequentially (no shuffle).
            loss_fn_id: The id_at_location of (the ObjectWrapper of) a loss function which
                        shall be used to calculate the loss. This is used internally for train config deserialization.
            model_id: id_at_location of a traced torch nn.Module instance (objectwrapper). . This is used internally for train config deserialization.
        """
        # syft related attributes
        self.owner = owner if owner else sy.hook.local_worker
        self.id = id if id is not None else sy.ID_PROVIDER.pop()
        self.location = None

        # training related attributes
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = lr
        self.max_nr_batches = max_nr_batches
        self.shuffle = shuffle

        # pointers
        self.model_ptr = None
        self.loss_fn_ptr = None

        # internal ids
        self._model_id = model_id
        self._loss_fn_id = loss_fn_id

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

    def _wrap_and_send_jit_module(self, module, location):
        """Wrappers jit module and send it to location."""
        module_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=module)
        module_ptr = self.owner.send(module_with_id, location)
        module_id = module_ptr.id_at_location
        return module_ptr, module_id

    def send(self, location: workers.BaseWorker) -> weakref:
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
        # Send traced model
        self.model_ptr, self._model_id = self._wrap_and_send_jit_module(self.model, location)

        # Send loss function
        self.loss_fn_ptr, self._loss_fn_id = self._wrap_and_send_jit_module(self.loss_fn, location)

        # Send train configuration itself
        ptr = self.owner.send(self, location)

        return ptr

    def get(self, location):
        return self.owner.request_obj(self, location)

    def get_model(self):
        if self.model is not None:
            return self.model_ptr.get()

    def get_loss_fn(self):
        if self.loss_fn is not None:
            return self.loss_fn.get()
