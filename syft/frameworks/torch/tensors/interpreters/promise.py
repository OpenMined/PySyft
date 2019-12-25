import syft as sy
from syft.workers.abstract import AbstractWorker
import weakref

from syft.generic.tensor import AbstractTensor
from syft.generic.tensor import initialize_tensor
from syft.messaging.promise import Promise
from syft.generic.frameworks.hook import hook_args


class PromiseTensor(AbstractTensor, Promise):
    def __init__(
        self, shape, owner=None, id=None, tensor_type=None, plans=None, tags=None, description=None
    ):
        """Initializes a PromiseTensor

        Args:
            shape: the shape that should have the tensors keeping the promise.
            owner: an optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: an optional string or integer id of the PromiseTensor.
            tensor_type: the type that should have the tensors keeping the promise.
            plans: the ids of the plans waiting for the promise to be kept. When the promise is
                kept, all the plans corresponding to these ids will be executed if the other
                promises they were waiting for are also kept.
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for.
            description: an optional string describing the purpose of the
                tensor.
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
    def simplify(worker: AbstractWorker, tensor: "PromiseTensor") -> tuple:
        """Takes the attributes of a PromiseTensor and saves them in a tuple.

        Args:
            tensor: a PromiseTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the Promise tensor.
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, tensor.id),
            sy.serde.msgpack.serde._simplify(worker, tensor.shape),
            sy.serde.msgpack.serde._simplify(worker, tensor.obj_type),
            sy.serde.msgpack.serde._simplify(worker, tensor.plans),
            sy.serde.msgpack.serde._simplify(worker, tensor.tags),
            sy.serde.msgpack.serde._simplify(worker, tensor.description),
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PromiseTensor":
        """
            This function reconstructs a PromiseTensor given it's attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the PromiseTensor
            Returns:
                PromiseTensor: a PromiseTensor
            Examples:
                shared_tensor = detail(data)
            """

        id, shape, tensor_type, plans, tags, description = tensor_tuple

        id = sy.serde.msgpack.serde._detail(worker, id)
        shape = sy.serde.msgpack.serde._detail(worker, shape)
        tensor_type = sy.serde.msgpack.serde._detail(worker, tensor_type)
        plans = sy.serde.msgpack.serde._detail(worker, plans)
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)

        tensor = PromiseTensor(
            owner=worker,
            id=id,
            shape=shape,
            tensor_type=tensor_type,
            plans=plans,
            tags=tags,
            description=description,
        )

        return tensor


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PromiseTensor)
