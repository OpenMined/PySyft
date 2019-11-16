from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.workers.abstract import AbstractWorker
import syft as sy
import numpy as np
import torch as th


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
        print("creating paillier tensor 2")

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

        new_child = list()
        for x in self.child.flatten().tolist():
            new_child.append(public_key.encrypt(x))
        data = np.array(new_child).reshape(self.child.shape)
        self.child = data

    def decrypt(self, private_key):
        """This method will decrypt each value in the tensor, returning a normal
              torch tensor.

              Args:
                  *private_key a private key created using
                      syft.frameworks.torch.he.paillier.keygen()
                  """
        new_child = list()
        for x in self.child.flatten().tolist():
            new_child.append(private_key.decrypt(x))

        return th.tensor(new_child).view(*self.child.shape)

    # Method overloading
    @overloaded.method
    def __add__(self, _self, *args, **kwargs):
        """
        Here is an example of how to use the @overloaded.method decorator. To see
        what this decorator do, just look at the next method manual_add: it does
        exactly the same but without the decorator.

        Note the subtlety between self and _self: you should use _self and NOT self.
        """
        print("Log method __add__")
        response = getattr(_self, "__add__")(*args, **kwargs)

        return response

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
            chain = sy.serde._simplify(worker, tensor.child)
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
            chain = sy.serde._detail(worker, chain)
            tensor.child = chain

        return tensor
