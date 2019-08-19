import syft as sy
import torch as th
import weakref

from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.messaging.promise import Promise


class PromiseTensor(AbstractTensor, Promise):
    def __init__(
        self,
        shape,
        owner=None,
        id=None,
        tags=None,
        description=None,
        tensor_id=None,
        tensor_type=None,
        plans=None,
    ):
        """Initializes a PromiseTensor

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LoggingTensor.
        """

        # I did check that the two __init__ methods below only get called once, but it
        # was exhibiting some strange behavior when I used super() for both of them.

        # constructor for AbstractTensor
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        # constructor for Promise
        Promise.__init__(self, obj_id=tensor_id, obj_type=tensor_type, plans=plans)

        self._shape = shape

    def torch_type(self):
        return self.obj_type

    @property
    def shape(self):
        return self._shape

    def on(self, tensor: "AbstractTensor", wrap: bool = True) -> "AbstractTensor":
        """
        Add a syft(log) tensor on top of the tensor.

        Args:
            tensor: the tensor to extend
            wrap: if true, add the syft tensor between the wrapper
            and the rest of the chain. If false, just add it at the top

        Returns:
            a syft/torch tensor
        """

        # This is the only difference from AbstractTensor.on()
        self.obj_type = tensor.type()

        if not wrap:

            self.child = tensor

            return self

        else:

            # if tensor is a wrapper
            if not hasattr(tensor, "child"):
                tensor = tensor.wrap()

            self.child = tensor.child
            tensor.child = self

            tensor.child.parent = weakref.ref(tensor)
            return tensor

def CreatePromiseTensor(shape, tensor_type: str, *args, **kwargs):
    return PromiseTensor(shape, *args, tensor_type=tensor_type, **kwargs).wrap()


class Promises:
    @staticmethod
    def FloatTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.FloatTensor", *args, **kwargs)

    @staticmethod
    def DoubleTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.DoubleTensor", *args, **kwargs)

    @staticmethod
    def HalfTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.HalfTensor", *args, **kwargs)

    @staticmethod
    def ByteTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.ByteTensor", *args, **kwargs)

    @staticmethod
    def CharTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.CharTensor", *args, **kwargs)

    @staticmethod
    def ShortTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.ShortTensor", *args, **kwargs)

    @staticmethod
    def IntTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.IntTensor", *args, **kwargs)

    @staticmethod
    def LongTensor(shape, *args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.LongTensor", *args, **kwargs)

    @staticmethod
    def BoolTensor(shape, args, **kwargs):
        return CreatePromiseTensor(shape, tensor_type="torch.BoolTensor", *args, **kwargs)
