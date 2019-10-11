import functools
from typing import List
import weakref

import syft as sy
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.object import _apply_args
from syft.generic.object import AbstractObject
from syft.generic.object import initialize_object


class AbstractTensor(AbstractObject):
    def __init__(
        self,
        id: int = None,
        owner: "sy.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
        child=None,
    ):
        super(AbstractTensor, self).__init__(id, owner, tags, description, child)

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
        if not wrap:
            self.child = tensor
            return self
        else:
            # if tensor is a wrapper
            if not hasattr(tensor, "child"):
                tensor = tensor.wrap()

            # We usually call .on() on newly created tensor so it's not a sacrilege
            # to rewrite its id
            self.id = tensor.id

            self.child = tensor.child
            tensor.child = self
            return tensor

    def copy(self):
        return self + 0

    def refresh(self):
        """
        Forward to Additive Shared Tensor the call to refresh shares
        """
        if hasattr(self, "child"):
            self.child = self.child.refresh()
            return self
        else:
            raise AttributeError("Refresh should only be called on AdditiveSharedTensors")

    @property
    def shape(self):
        return self.child.shape

    def __len__(self) -> int:
        """Alias .shape[0] with len(), helpful for pointers"""
        try:
            if hasattr(self, "child") and not isinstance(self.child, dict):
                return self.child.shape[0]
            else:
                return self.shape[0]
        except IndexError:
            return 0

    @property
    def grad(self):
        child_grad = self.child.grad
        if child_grad is None:
            return None
        else:
            return child_grad.wrap()


def initialize_tensor(hook, obj, owner=None, id=None, init_args=tuple(), init_kwargs={}):
    """Initializes the tensor.

    Args:
        hook: A reference to TorchHook class.
        cls: An object to keep track of id, owner and whether it is a native
            tensor or a wrapper over pytorch.
        is_tensor: A boolean parameter (default False) to indicate whether
            it is torch tensor or not.
        owner: The owner of the tensor being initialised, leave it blank
            to if you have already provided a reference to TorchHook class.
        id: The id of tensor, a random id will be generated if there is no id
            specified.
    """
    initialize_object(
        hook,
        obj,
        owner=owner,
        reinitialize=False,
        id=id,
        init_args=init_args,
        init_kwargs=init_kwargs,
    )
