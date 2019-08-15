import syft as sy
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded
from syft.messaging.promise import Promise


class PromiseTensor(AbstractTensor, Promise):
    def __init__(self, owner=None, id=None, tags=None, description=None, tensor_id=None, plans=None):
        """Initializes a LoggingTensor, whose behaviour is to log all operations
        applied on it.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LoggingTensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description, tensor_id=tensor_id, plans=plans)

    # Method overloading

    @overloaded.method
    def add(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """
        print("Log method add")
        response = getattr(_self, "add")(*args, **kwargs)

        return response