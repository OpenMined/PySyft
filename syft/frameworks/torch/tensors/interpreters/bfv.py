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

from syft.frameworks.torch.he.fv.decryptor import Decryptor
from syft.frameworks.torch.he.fv.encryptor import Encryptor
from syft.frameworks.torch.he.fv.integer_encoder import IntegerEncoder
from syft.frameworks.torch.he.fv.evaluator import Evaluator

# from syft.frameworks.torch.he.fv.plaintext import PlainText
# from syft.frameworks.torch.he.fv.ciphertext import CipherText


class BFVTensor(AbstractTensor):
    def __init__(self, **kwargs):
        """Initializes a BFVTensor.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the BFVTensor.
        """
        super().__init__(**kwargs)
        self.context = None
        self.encryptor = None
        self.decryptor = None
        self.encoder = None
        self.evaluator = None

    def encrypt(self, key, context):
        self.context = context
        self.prepareEncryptor(key)

        output = BFVTensor()
        output.child = self.child
        inputs = self.child.flatten().tolist()
        new_child = [self.encryptor.encrypt(self.encoder.encode(x)) for x in inputs]

        data = np.array(new_child).reshape(self.child.shape)
        self.child = data
        return output

    def decrypt(self, private_key):
        self.prepareDecryptor(private_key)

        inputs = self.child.flatten().tolist()
        new_child = [self.encoder.decode(self.decryptor.decrypt(x)) for x in inputs]
        return th.tensor(new_child).view(*self.child.shape)

    def __add__(self, *args, **kwargs):
        self.prepareEvaluator()

        if not args[0].shape == self.child.shape:
            raise ValueError(
                f"Cannot add tensors with different shape {args[0].shape} , {self.child.shape}"
            )

        args = args[0].flatten().tolist()

        # if integer values; transform it to plaintext
        if isinstance(args[0], int):
            args = [self.encoder.encode(x) for x in args]

        child = self.child.flatten().tolist()

        data = [self.evaluator.add(x, y) for x, y in zip(child, args)]
        data = np.array(data).reshape(self.child.shape)
        obj = BFVTensor()
        obj.context = self.context
        obj.child = data
        return obj

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
        self.prepareEvaluator()

        if not args[0].shape == self.child.shape:
            raise ValueError(
                f"Cannot subtract tensors with different shape {args[0].shape} , {self.child.shape}"
            )

        args = args[0].flatten().tolist()

        # if integer values; transform it to plaintext
        if isinstance(args[0], int):
            args = [self.encoder.encode(x) for x in args]

        child = self.child.flatten().tolist()

        data = [self.evaluator.sub(x, y) for x, y in zip(child, args)]
        data = np.array(data).reshape(self.child.shape)
        obj = BFVTensor()
        obj.context = self.context
        obj.child = data
        return obj

        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
            "__add__", self, args, kwargs
        )

        # Send it to the appropriates class and get the response
        response = getattr(new_self, "__add__")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response("__add__", response, wrap_type=type(self))
        return response

    def __mul__(self, *args, **kwargs):
        print("custom mul called!")
        pass

    def mm(self, *args, **kwargs):
        print("custom mm called!")
        pass

    # Method overloading
    @overloaded.method
    def add(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """

        return _self + args[0]

    # Method overloading
    @overloaded.method
    def sub(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """

        return _self - args[0]

    # Method overloading
    @overloaded.method
    def mul(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """

        return _self * args[0]

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
            pass

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
            pass

        # Just register it using the module variable
        module.mul = mul

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "BFVTensor") -> tuple:
        """
        This function takes the attributes of a BFVTensor and saves them in a tuple
        Args:
            tensor (BFVTensor): # TODO
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
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "BFVTensor":
        """
        This function reconstructs a BFVTensor given it's attributes in form of a tuple.
        Args:
            worker: the worker doing the deserialization
            tensor_tuple: a tuple holding the attributes of the BFVTensor
        Returns:
            BFVTensor: # TODO
        """
        obj_id, chain = tensor_tuple

        tensor = BFVTensor(owner=worker, id=obj_id)

        if chain is not None:
            chain = sy.serde.msgpack.serde._detail(worker, chain)
            tensor.child = chain

        return tensor

    def prepareEncryptor(self, key):
        if self.encoder is None:
            self.encoder = IntegerEncoder(self.context)
        if self.encryptor is None:
            self.encryptor = Encryptor(self.context, key)

    def prepareDecryptor(self, private_key):
        if self.encoder is None:
            self.encoder = IntegerEncoder(self.context)
        if self.decryptor is None:
            self.decryptor = Decryptor(self.context, private_key)

    def prepareEvaluator(self):
        if self.encoder is None:
            self.encoder = IntegerEncoder(self.context)
        if self.evaluator is None:
            self.evaluator = Evaluator(self.context)


register_type_rule({BFVTensor: one})
register_forward_func({BFVTensor: get_child})
register_backward_func({BFVTensor: lambda i, **kwargs: BFVTensor().on(i, wrap=False)})
