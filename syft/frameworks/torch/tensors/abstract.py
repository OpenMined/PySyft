from abc import ABC, abstractmethod
import torch
import random
import syft as sy


class AbstractTensor(ABC):
    """
    This is the tensor abstraction.
    """

    def wrap(self):
        wrapper = torch.Tensor()
        wrapper.child = self
        wrapper.is_wrapper = True
        return wrapper

    def serialize(self,
                  compress=True,
                  compress_scheme=0): # Code 0 is LZ4 - check serde.py to see others
        """This convenience method serializes the tensor on which it's called.

            This is the high level convenience function for serializing torch tensors.
            It includes three steps, Simplify, Serialize, and Compress as described
            in serde.py

            Args:
                self (AbstractTensor): the tensor to be serialized

                compress (bool): whether or not to compress the object

                compress_scheme (int): the integer code specifying which compression
                    scheme to use (see serde.py for scheme codes) if compress == True.
                    The compression scheme is set to LZ4 by default (code 0).

            Returns:
                binary: the serialized form of the tensor.

            Examples:

                x = torch.Tensor([1,2,3,4,5])
                x.serialize() # returns a serialized object
        """

        return sy.serde.serialize(self,
                                  compress=compress,
                                  compress_scheme=compress_scheme)



def initialize_tensor(
    hook_self, cls, torch_tensor: bool = False, owner=None, id=None, *init_args, **init_kwargs
):

    cls.is_wrapper = False

    if not torch_tensor:
        cls.native___init__(*init_args, **init_kwargs)

    if owner is None:
        owner = hook_self.local_worker

    if id is None:
        id = int(10e10 * random.random())

    cls.id = id
    cls.owner = owner
