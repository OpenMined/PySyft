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

        # constructors for AbstractTensor and Promise
        AbstractTensor.__init__(self, id=id, owner=owner, tags=tags, description=description)
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
            sy.serde._simplify(self.shape),
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

        return tensor


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PromiseTensor)
