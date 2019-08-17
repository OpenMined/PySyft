import syft as sy
import torch as th

from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded
from syft.messaging.promise import Promise


class PromiseTensor(AbstractTensor, Promise):
    def __init__(
        self, owner=None, id=None, tags=None, description=None, tensor_id=None, plans=None
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
        super().__init__(
            id=id, owner=owner, tags=tags, description=description
        )

        # constructor for Promise
        Promise.__init__(self, obj_id=tensor_id, plans=plans)


    def __add__(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you might need sometimes to specify
        some particular behaviour: so here what to start from :)
        """

        other = args[0]

        print(self)
        print(other)

        @sy.func2plan([th.tensor([1]), th.tensor([2])])
        def add(a, b):
            return a + b

        add.arg_ids = [self.obj_id, other.obj_id]

        self.plans.add(add)
        other.plans.add(add)

        # only need this for use of Promises with the local_worker VirtualWorker
        # otherwise we would simplty check the ._objects registry
        add.args_fulfilled = {}

        self.result_promise = PromiseTensor(tensor_id=add.result_ids[0], plans=set())
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
