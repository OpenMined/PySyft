from abc import ABC
import torch
import random
import syft as sy


class AbstractTensor(ABC):
    """This is the tensor abstraction.
    """

    def wrap(self):
        """Wraps the class inside torch tensor.

        Returns:
            A pytorch tensor.
        """
        wrapper = torch.Tensor()
        wrapper.child = self
        wrapper.is_wrapper = True
        return wrapper

    def serialize(self, compress=True, compress_scheme=0):
        """Serializes the tensor on which it's called.

        This is the high level convenience function for serializing torch
        tensors. It includes three steps, Simplify, Serialize, and Compress as
        described in serde.py.

        Args:
            compress (bool): Whether or not to compress the object
            compress_scheme (int): Integer code specifying the compression
                scheme to use (see serde.py for scheme codes) if compress is
                True. The compression scheme is set to LZ4 by default (code 0).

        Returns:
            The serialized form of the tensor.
            For example:
                x = torch.Tensor([1,2,3,4,5])
                x.serialize() # returns a serialized object
        """
        # Code 0 is LZ4 - check serde.py to see others
        return sy.serde.serialize(
            self, compress=compress, compress_scheme=compress_scheme)


def initialize_tensor(
    hook_self, cls, torch_tensor: bool = False,
    owner=None, id=None, *init_args, **init_kwargs
):
    """Initializes the tensor.

    Args:
        hook_self: A reference to TorchHook class.
        cls: An object to keep track of id, owner and whether it is a native
            tensor or a wrapper over pytorch.
        torch_tensor (bool): Whether or not it is torch tensor.
        owner: The owner of the tensor being initialised, leave it blank
            to if you have already provided a reference to TorchHook class.
        id: The id of tensor, a random id will be generated if there is no id
            specified.
    """
    cls.is_wrapper = False
    if not torch_tensor:
        cls.native___init__(*init_args, **init_kwargs)
    if owner is None:
        owner = hook_self.local_worker
    if id is None:
        id = int(10e10 * random.random())
    cls.id = id
    cls.owner = owner
