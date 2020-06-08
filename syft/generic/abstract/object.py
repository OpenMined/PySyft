from abc import ABC
import functools
from typing import Set

import syft as sy
from syft.generic.frameworks.hook import hook_args


class AbstractObject(ABC):
    """
    This is a generic object abstraction.
    """

    is_wrapper = False

    def __init__(
        self,
        id: int = None,
        owner: "sy.workers.AbstractWorker" = None,
        tags: Set[str] = None,
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
        self.owner = owner or sy.local_worker
        self.id = id or sy.ID_PROVIDER.pop()
        self.tags = tags or set()
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

    def describe(self, description: str) -> "AbstractObject":
        self.description = description
        return self

    def tag(self, *tags: str) -> "AbstractObject":
        self.tags = self.tags or set()

        for tag in tags:
            self.tags.add(tag)

        self.owner.object_store.register_tags(self)

        return self

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
        Replace in the args_ all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a Syft Tensor on top of all tensors found in
        the response.

        Args:
            command: instruction of a function command: (command name,
            <no self>, arguments[, kwargs_])

        Returns:
            the response of the function command
        """
        cmd, _, args_, kwargs_ = command

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args_, **kwargs_)
        except AttributeError:
            pass

        # Replace all LoggingTensor with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args_, kwargs_)

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Do a generic action depending og the call
        cls.on_function_call(new_command)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back LoggingTensor on the tensors found in the response
        response = hook_args.hook_response(cmd, response, wrap_type=cls)

        return response

    @classmethod
    def rgetattr(cls, obj, attr, *args):
        """
        Get an attribute recursively.

        This is a core piece of functionality for the PySyft tensor chain.

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


def initialize_object(
    hook, obj, owner=None, reinitialize=True, id=None, init_args=(), init_kwargs={}
):
    """Initializes the tensor.

    Args:
        hook: A reference to TorchHook class.
        obj: An object to keep track of id, owner and whether it is a native
            tensor or a wrapper over pytorch.
        reinitialize: A boolean parameter (default True) to indicate whether
            to re-execute __init__.
        owner: The owner of the tensor being initialised, leave it blank
            to if you have already provided a reference to TorchHook class.
        id: The id of tensor, a random id will be generated if there is no id
            specified.
    """
    obj.is_wrapper = False

    if reinitialize:
        obj.native___init__(*init_args, **init_kwargs)

    _apply_args(hook, obj, owner, id)


def _apply_args(hook, obj_to_register, owner=None, id=None):

    if owner is None:
        owner = hook.local_worker

    if id is None:
        id = sy.ID_PROVIDER.pop()

    obj_to_register.id = id
    obj_to_register.owner = owner
