from abc import ABC
import functools
import torch
from typing import List

import syft as sy
import weakref


class AbstractObject(ABC):
    """
    This is a generic object abstraction.
    """

    is_wrapper = False

    def __init__(
        self,
        id: int = None,
        owner: "sy.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
        child=None,
    ):
        """Initializer for AbstractTensor

        Args:
            id: An optional string or integer id of the tensor
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
            child: an optional tensor to put in the .child attribute to build
                a chain of tensors
        """
        self.owner = owner
        if id is None:
            self.id = sy.ID_PROVIDER.pop()
        else:
            self.id = id
        self.tags = tags
        self.description = description
        self.child = child

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
    def shape(self):
        return self.child.shape

    def serialize(self):  # check serde.py to see how to provide compression schemes
        """Serializes the tensor on which it's called.

        This is the high level convenience function for serializing torch
        tensors. It includes three steps, Simplify, Serialize, and Compress as
        described in serde.py.
        By default serde is compressing using LZ4

        Returns:
            The serialized form of the tensor.
            For example:
                x = torch.Tensor([1,2,3,4,5])
                x.serialize() # returns a serialized object
        """
        return sy.serde.serialize(self)

    def ser(self, *args, **kwargs):
        return self.serialize(*args, **kwargs)

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        Syft tensor which has a child which is a pointer, an additive shared tensor,
        a multi-pointer, etc."""
        class_attributes = self.get_class_attributes()
        return type(self)(
            **class_attributes,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
            id=self.id,
        ).on(self.child.get())

    def mid_get(self):
        """This method calls .get() on a child pointer and correctly registers the results"""

        child_id = self.id
        tensor = self.get()
        tensor.id = child_id
        self.owner.register_obj(tensor)

    def get_class_attributes(self):
        """
        Return all elements which defines an instance of a certain class.
        By default there is nothing so we return an empty dict, but for
        example for fixed precision tensor, the fractional precision is
        very important.
        """
        return {}

    @classmethod
    def on_function_call(cls, *args):
        """
        Override this to perform a specific action for each call of a torch
        function with arguments containing syft tensors of the class doing
        the overloading
        """
        pass

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Syft Tensor,
        Replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a Syft Tensor on top of all tensors found in
        the response.

        Args:
            command: instruction of a function command: (command name,
            <no self>, arguments[, kwargs])

        Returns:
            the response of the function command
        """
        cmd, _, args, kwargs = command

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all LoggingTensor with their child attribute
        new_args, new_kwargs, new_type = sy.frameworks.torch.hook_args.unwrap_args_from_function(
            cmd, args, kwargs
        )

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Do a generic action depending og the call
        cls.on_function_call(new_command)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back LoggingTensor on the tensors found in the response
        response = sy.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)

        return response

    @classmethod
    def rgetattr(cls, obj, attr, *args):
        """
        Get an attribute recursively


        Args:
            obj: the object holding the attribute
            attr: nested attribute
            args: optional arguments to provide

        Returns:
            the attribute obj.attr

        Example:
            >>> rgetattr(obj, 'attr1.attr2.attr3')
            [Out] obj.attr1.attr2.attr3

        """

        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))


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

            self.child = tensor.child
            tensor.child = self
            return tensor

    def wrap(self) -> torch.Tensor:
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
        wrapper.child.parent = weakref.ref(wrapper)

        if self.id is None:
            self.id = sy.ID_PROVIDER.pop()

        return wrapper

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
    def grad(self):
        child_grad = self.child.grad
        if child_grad is None:
            return None
        else:
            return child_grad.wrap()


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
        id = sy.ID_PROVIDER.pop()

    new_tensor.id = id
    new_tensor.owner = owner
