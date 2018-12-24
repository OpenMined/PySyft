from abc import ABC, abstractmethod
import torch
import random
import syft as sy


class AbstractTensor(ABC):
    """
    This is the tensor abstraction
    """

    def wrap(self):
        wrapper = torch.Tensor()
        wrapper.child = self
        wrapper.is_wrapper = True
        return wrapper

    def serialize(self,
                  compress=True,
                  compress_scheme="lz4"):

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
