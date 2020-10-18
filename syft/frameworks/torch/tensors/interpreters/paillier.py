import syft as sy
import numpy as np
import torch as th

from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.hook.hook_args import (
    get_child,
    register_backward_func,
    register_forward_func,
    register_type_rule,
    one,
)
from syft.generic.frameworks.overload import overloaded
from syft.workers.abstract import AbstractWorker


class PaillierTensor(AbstractTensor):
    def __init__(self, owner=None, id=None, tags=None, description=None):
        """Initializes a PaillierTensor, whose behaviour is to log all operations
        applied on it.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the PaillierTensor.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

    def encrypt(self, public_key):
        """This method will encrypt each value in the tensor using Paillier
        homomorphic encryption.

        Args:
            *public_key a public key created using
                syft.frameworks.torch.he.paillier.keygen()
        """

        output = PaillierTensor()
        output.child = self.child
        output.encrypt_(public_key)
        return output

    def encrypt_(self, public_key):
        """This method will encrypt each value in the tensor using Paillier
        homomorphic encryption.

        Args:
            *public_key a public key created using
                syft.frameworks.torch.he.paillier.keygen()
        """

        inputs = self.child.flatten().tolist()
        new_child = sy.pool().map(public_key.encrypt, inputs)

        data = np.array(new_child).reshape(self.child.shape)
        self.child = data
        self.pubkey = public_key

    def decrypt(self, private_key):
        """This method will decrypt each value in the tensor, returning a normal
        torch tensor.

        =Args:
            *private_key a private key created using
                syft.frameworks.torch.he.paillier.keygen()
        """

        if not isinstance(self.child, np.ndarray):
            return th.tensor(private_key.decrypt(self.child))

        inputs = self.child.flatten().tolist()

        new_child = sy.pool().map(private_key.decrypt, inputs)

        return th.tensor(new_child).view(*self.child.shape)

    def __add__(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you misght need sometimes to specify
        some particular behaviour: so here what to start from :)
        """

        if isinstance(args[0], th.Tensor):
            data = self.child + args[0].numpy()
            obj = PaillierTensor()
            obj.child = data
            return obj

        if isinstance(self.child, th.Tensor):
            self.child = self.child.numpy()

        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__add__", self, args, kwargs
        )

        # Send it to the appropriates class and get the response
        response = getattr(new_self, "__add__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__add__", response, wrap_type=type(self))
        return response

    def __sub__(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you misght need sometimes to specify
        some particular behaviour: so here what to start from :)
        """

        if isinstance(args[0], th.Tensor):
            data = self.child - args[0].numpy()
            obj = PaillierTensor()
            obj.child = data
            return obj

        if isinstance(self.child, th.Tensor):
            self.child = self.child.numpy()

        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__sub__", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "__sub__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__sub__", response, wrap_type=type(self))
        return response

    def __mul__(self, *args, **kwargs):
        """
        Here is the version of the add method without the decorator: as you can see
        it is much more complicated. However you misght need sometimes to specify
        some particular behaviour: so here what to start from :)
        """

        if isinstance(args[0], th.Tensor):
            data = self.child * args[0].numpy()
            obj = PaillierTensor()
            obj.child = data
            return obj

        if isinstance(self.child, th.Tensor):
            self.child = self.child.numpy()

        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__mul__", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "__mul__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__mul__", response, wrap_type=type(self))
        return response

    def mm(self, *args, **kwargs):
        """
        Here is matrix multiplication between an encrypted and unencrypted tensor. Note that
        we cannot matrix multiply two encrypted tensors because Paillier does not support
        the multiplication of two encrypted values.
        """
        out = PaillierTensor()

        # if self is not encrypted and args[0] is encrypted
        if isinstance(self.child, th.Tensor):
            out.child = self.child.numpy().dot(args[0].child)

        # if self is encrypted and args[0] is not encrypted
        else:
            out.child = self.child.dot(args[0])

        return out

    # Method overloading
    @overloaded.method
    def add(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """

        return self + args[0]

    # Method overloading
    @overloaded.method
    def sub(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """

        return self - args[0]

    # Method overloading
    @overloaded.method
    def mul(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """

        return self * args[0]

    # Module & Function overloading

    # We overload two torch functions:
    # - torch.add
    # - torch.nn.functional.relu

    @staticmethod
    @overloaded.module
    def torch(module):
        """
        We use the @overloaded.module to specify we're writing here
        a function which should overload the function with the same
        name in the <torch> module
        :param module: object which stores the overloading functions

        Note that we used the @staticmethod decorator as we're in a
        class
        """

        def add(x, y):
            """
            You can write the function to overload in the most natural
            way, so this will be called whenever you call torch.add on
            Logging Tensors, and the x and y you get are also Logging
            Tensors, so compared to the @overloaded.method, you see
            that the @overloaded.module does not hook the arguments.
            """
            print("Log function torch.add")
            return x + y

        # Just register it using the module variable
        module.add = add

        def mul(x, y):
            """
            You can write the function to overload in the most natural
            way, so this will be called whenever you call torch.add on
            Logging Tensors, and the x and y you get are also Logging
            Tensors, so compared to the @overloaded.method, you see
            that the @overloaded.module does not hook the arguments.
            """
            print("Log function torch.mul")
            return x * y

        # Just register it using the module variable
        module.mul = mul

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "PaillierTensor") -> tuple:
        """
        This function takes the attributes of a LogTensor and saves them in a tuple
        Args:
            tensor (PaillierTensor): a LogTensor
        Returns:
            tuple: a tuple holding the unique attributes of the log tensor
        Examples:
            data = _simplify(tensor)
        """

        chain = None
        if hasattr(tensor, "child"):
            chain = sy.serde.msgpack.serde._simplify(worker, tensor.child)
        return tensor.id, chain

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PaillierTensor":
        """
        This function reconstructs a LogTensor given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the LogTensor
        Returns:
            PaillierTensor: a LogTensor
        Examples:
            logtensor = detail(data)
        """
        obj_id, chain = tensor_tuple

        tensor = PaillierTensor(owner=worker, id=obj_id)

        if chain is not None:
            chain = sy.serde.msgpack.serde._detail(worker, chain)
            tensor.child = chain

        return tensor


register_type_rule({PaillierTensor: one})
register_forward_func({PaillierTensor: get_child})
register_backward_func({PaillierTensor: lambda i, **kwargs: PaillierTensor().on(i, wrap=False)})
