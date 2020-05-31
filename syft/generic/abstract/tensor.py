from typing import List
import weakref

import syft as sy
from syft.generic.abstract.object import _apply_args  # noqa: F401
from syft.generic.abstract.object import initialize_object
from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.abstract import AbstractWorker
from syft.serde.syft_serializable import SyftSerializable


class AbstractTensor(AbstractSendable, SyftSerializable):
    def __init__(
        self,
        id: int = None,
        owner: "sy.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
        child=None,
    ):
        super(AbstractTensor, self).__init__(id, owner, tags, description, child)

    def wrap(self, register=True, type=None, **kwargs):
        """Wraps the class inside an empty object of class `type`.

        Because PyTorch/TF do not (yet) support functionality for creating
        arbitrary Tensor types (via subclassing torch.Tensor), in order for our
        new tensor types (such as PointerTensor) to be usable by the rest of
        PyTorch/TF (such as PyTorch's layers and loss functions), we need to
        wrap all of our new tensor types inside of a native PyTorch type.

        This function adds a .wrap() function to all of our tensor types (by
        adding it to AbstractTensor), such that (on any custom tensor
        my_tensor), my_tensor.wrap() will return a tensor that is compatible
        with the rest of the PyTorch/TensorFlow API.

        Returns:
            A wrapper tensor of class `type`, or whatever is specified as
            default by the current syft.framework.Tensor.
        """
        wrapper = sy.framework.hook.create_wrapper(type, **kwargs)
        wrapper.child = self
        wrapper.is_wrapper = True
        wrapper.child.parent = weakref.ref(wrapper)

        if self.id is None:
            self.id = sy.ID_PROVIDER.pop()

        if self.owner is not None and register:
            self.owner.register_obj(wrapper, obj_id=self.id)

        return wrapper

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

    def clone(self):
        """
        Clone should keep ids unchanged, contrary to copy
        """
        cloned_tensor = type(self)(**self.get_class_attributes())
        cloned_tensor.id = self.id
        cloned_tensor.owner = self.owner

        if hasattr(self, "child") and self.child is not None:
            cloned_tensor.child = self.child.clone()

        return cloned_tensor

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

    def send_(self, *location, **kwargs):
        """
        Calls send() with inplace option, but only with a single location
        :param location: workers locations
        :return:
        """
        if len(location) > 1:
            raise NotImplementedError("Inplace send to several workers is currently not supported.")

        return self.send(*location, inplace=True, **kwargs)

    def get_(self, *args, **kwargs):
        """
        Calls get() with inplace option set to True
        """
        return self.get(*args, inplace=True, **kwargs)

    def allow(self, user=None) -> bool:
        """ This function returns will return True if it isn't a PrivateTensor, otherwise it will
        return the result of PrivateTensor's allow method.

            Args:
                user (object,optional): User credentials to be verified.

            Returns:
                boolean: If it is a public tensor/ allowed user, returns true, otherwise it returns
                false.
        """
        # If it is a wrapper
        if self.is_wrapper:
            current_tensor = self.child

            # Verify permissions for each element on the tensor chain.
            while hasattr(current_tensor, "child"):

                # If it has a list of allowed users, verify permissions,
                # otherwise (public tensors) go to the next.
                if hasattr(current_tensor, "allowed_users"):
                    allow = current_tensor.allow(user)
                    if not allow:
                        return False

                # Go to next element on the tensor chain
                current_tensor = current_tensor.child
        return True

    def move_(self, location: AbstractWorker, requires_grad: bool = False):
        """
        Inplace version of move
        """
        new_ptr = self.move(location, requires_grad)
        self.child = new_ptr
        return self

    def remote_send(self, location):
        return self.child.remote_send(location).wrap()


def initialize_tensor(hook, obj, owner=None, id=None, init_args=(), init_kwargs={}):
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
