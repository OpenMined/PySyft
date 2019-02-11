from abc import ABC
import torch
import random
import syft as sy
import weakref


class AbstractTensor(ABC):
    """
    This is the tensor abstraction.
    """

    is_wrapper = False

    def __init__(self, tags=None, description=None):
        """Initializer for AbstractTensor

        Args:
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for.
            description: an optional string describing the purpose of the
                tensor
        """

        self.tags = tags
        self.description = description

    def __str__(self) -> str:
        if hasattr(self, "child"):
            return type(self).__name__ + ">" + self.child.__str__()
        else:
            return type(self).__name__

    def __repr__(self) -> str:
        if hasattr(self, "child"):
            return type(self).__name__ + ">" + self.child.__repr__()
        else:
            return type(self).__name__

    def on(self, tensor, wrap=True):
        """
        Add a syft(log) tensor on top of the tensor.
        :param tensor: the tensor to extend
        :param wrap: if true, add the syft tensor between the wrapper
        and the rest of the chain. If false, just add it at the top
        :return: a syft/torch tensor
        """
        if not wrap:
            self.child = tensor
            return self
        else:
            # if tensor is a wrapper
            if not hasattr(tensor, "child"):
                tensor = tensor.wrap()

            self.child = tensor.child
            tensor.child = self
            return tensor

    def wrap(self):
        """Wraps the class inside torch tensor.

        Because PyTorch does not (yet) support functionality for creating
        arbitrary Tensor types (via subclassing torch.Tensor), in order for our
        new tensor types (such as PointerTensor) to be usable by the rest of
        PyTorch (such as PyTorch's layers and loss functions), we need to wrap
        all of our new tensor types inside of a native PyTorch type.

        This function adds a .wrap() function to all of our tensor types (by
        adding it to AbstractTensor), such that (on any custom tensor
        my_tensor), my_tensor.wrap() will return a tensor that is compatible
        with the rest of the PyTorch API.

        Returns:
            A pytorch tensor.
        """
        wrapper = torch.Tensor()
        wrapper.child = self
        wrapper.is_wrapper = True
        # wrapper.child.parent = weakref.ref(wrapper)
        return wrapper

    def serialize(
        self, compress=True, compress_scheme=0
    ):  # Code 0 is LZ4 - check serde.py to see others
        """Serializes the tensor on which it's called.

        This is the high level convenience function for serializing torch
        tensors. It includes three steps, Simplify, Serialize, and Compress as
        described in serde.py.

        Args:
            compress: A boolean indicating whether to compress the object or
                not.
            compress_scheme: An integer code specifying the compression scheme
                to use (see serde.py for scheme codes) if compress is True. The
                compression scheme is set to LZ4 by default (code 0).

        Returns:
            The serialized form of the tensor.
            For example:
                x = torch.Tensor([1,2,3,4,5])
                x.serialize() # returns a serialized object
        """
        return sy.serde.serialize(self, compress=compress, compress_scheme=compress_scheme)

    def ser(self, *args, **kwargs):
        return self.serialize(*args, **kwargs)

    @property
    def shape(self):
        return self.child.shape

    def get_class_attributes(self):
        """
        Return all elements which defines an instance of a certain class.
        By default there is nothing so we return an empty dict, but for
        example for fixed precision tensor, the fractional precision is
        very important.
        """
        return {}


def initialize_tensor(
    hook_self, cls, torch_tensor: bool = False, owner=None, id=None, *init_args, **init_kwargs
):
    """Initializes the tensor.

    Args:
        hook_self: A reference to TorchHook class.
        cls: An object to keep track of id, owner and whether it is a native
            tensor or a wrapper over pytorch.
        torch_tensor: A boolean parameter (default False) to indicate whether
            it is torch tensor or not.
        owner: The owner of the tensor being initialised, leave it blank
            to if you have already provided a reference to TorchHook class.
        id: The id of tensor, a random id will be generated if there is no id
            specified.
    """
    cls.is_wrapper = False

    if not torch_tensor:
        cls.native___init__(*init_args, **init_kwargs)

    _apply_args(hook_self, cls, owner, id)


def _apply_args(hook_self, new_tensor, owner=None, id=None):
    if owner is None:
        owner = hook_self.local_worker

    if id is None:
        id = int(10e10 * random.random())

    new_tensor.id = id
    new_tensor.owner = owner
