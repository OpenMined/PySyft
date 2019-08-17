import syft as sy
import torch as th
import weakref

from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.messaging.promise import Promise


class PromiseTensor(AbstractTensor, Promise):
    def __init__(
        self,
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

    def __add__(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you might need sometimes to specify
        some particular behaviour: so here what to start from :)
        """

        other = args[0]

        @sy.func2plan([th.tensor([1]), th.tensor([2])])
        def __add__(self, other):
            return self.__add__(other)

        __add__.arg_ids = [self.obj_id, other.obj_id]

        self.plans.add(__add__)
        other.plans.add(__add__)

        # only need this for use of Promises with the local_worker VirtualWorker
        # otherwise we would simplty check the ._objects registry
        __add__.args_fulfilled = {}

        self.result_promise = PromiseTensor(
            tensor_id=__add__.result_ids[0], tensor_type=self.obj_type, plans=set()
        )
        other.result_promise = self.result_promise

        return self.result_promise
        #
        # # Replace all syft tensor with their child attribute
        # new_self, new_args, new_kwargs = sy.frameworks.torch.hook_args.unwrap_args_from_method(
        #     "add", self, args, kwargs
        # )
        #
        # print("Log method manual_add")
        # # Send it to the appropriate class and get the response
        # response = getattr(new_self, "add")(*new_args, **new_kwargs)
        #
        # # Put back SyftTensor on the tensors found in the response
        # response = sy.frameworks.torch.hook_args.hook_response(
        #     "add", response, wrap_type=type(self)
        # )
        #
        # return response

    def torch_type(self):
        return self.obj_type

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


def CreatePromiseTensor(tensor_type:str, *args, **kwargs):
    return PromiseTensor(*args, tensor_type=tensor_type, **kwargs).wrap()


class Promises:
    @staticmethod
    def FloatTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.FloatTensor", *args, **kwargs)

    @staticmethod
    def DoubleTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.DoubleTensor", *args, **kwargs)

    @staticmethod
    def HalfTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.HalfTensor", *args, **kwargs)

    @staticmethod
    def ByteTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.ByteTensor", *args, **kwargs)

    @staticmethod
    def CharTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.CharTensor", *args, **kwargs)

    @staticmethod
    def ShortTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.ShortTensor", *args, **kwargs)

    @staticmethod
    def IntTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.IntTensor", *args, **kwargs)

    @staticmethod
    def LongTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.LongTensor", *args, **kwargs)

    @staticmethod
    def BoolTensor(*args, **kwargs):
        return CreatePromiseTensor(tensor_type="torch.BoolTensor", *args, **kwargs)
