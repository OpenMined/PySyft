import syft as sy
from syft.workers.abstract import AbstractWorker
import weakref

from syft.generic.tensor import AbstractTensor
from syft.generic.tensor import initialize_tensor
from syft.messaging.promise import Promise
from syft.generic.frameworks.hook import hook_args


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

        if owner is None:
            owner = sy.local_worker

        # constructor for AbstractTensor
        AbstractTensor.__init__(self, id=id, owner=owner, tags=tags, description=description)
        # constructor for Promise
        Promise.__init__(self, owner=owner, obj_type=tensor_type, plans=plans)

        self._shape = shape

        del self.child

    def torch_type(self):
        return self.obj_type

    @property
    def shape(self):
        return self._shape

    @property
    def grad(self):
        return None
        # if not hasattr(self, "_grad"):
        #     self._grad = PromiseTensor(shape=self._shape, tensor_type=self.torch_type()).wrap()
        #
        # return self._grad

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

    def __str__(self):
        return f"[PromiseTensor({self.owner.id}:{self.id}) -future-> {self.obj_type.split('.')[-1]} -blocking-> {len(self.plans)} plans]"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def simplify(self: "PromiseTensor") -> tuple:
        """Takes the attributes of a FixedPrecisionTensor and saves them in a tuple.

        Args:
            tensor: a FixedPrecisionTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed precision tensor.
        """

        return (
            sy.serde._simplify(self.id),
            sy.serde._simplify(self._shape),
            sy.serde._simplify(self.obj_type),
            sy.serde._simplify(self.plans),
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PromiseTensor":
        """
            This function reconstructs a FixedPrecisionTensor given it's attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the FixedPrecisionTensor
            Returns:
                FixedPrecisionTensor: a FixedPrecisionTensor
            Examples:
                shared_tensor = detail(data)
            """

        id, shape, tensor_type, plans = tensor_tuple

        id = sy.serde._detail(worker, id)
        shape = sy.serde._detail(worker, shape)
        tensor_type = sy.serde._detail(worker, tensor_type)
        plans = sy.serde._detail(worker, plans)

        tensor = PromiseTensor(
            owner=worker, id=id, shape=shape, tensor_type=tensor_type, plans=plans
        )

        initialize_tensor(
            hook_self=sy.torch.hook,
            cls=tensor,
            is_tensor=True,
            owner=worker,
            id=id,
            init_args=[],
            kwargs={},
        )

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


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PromiseTensor)
