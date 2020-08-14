import types
from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import List

import syft
from syft.exceptions import route_method_exception
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.frameworks.hook.pointers import PointerHook
from syft.generic.frameworks.hook.string import StringHook
from syft.generic.frameworks.hook.tensors import TensorHook
from syft.workers.base import BaseWorker


class FrameworkHook(TensorHook, PointerHook, StringHook, ABC):
    """Composite hook for ALL THE FRAMEWORK THINGS that must be overloaded and/or modified"""

    @abstractmethod
    def __init__(self, framework_module, local_worker: BaseWorker = None, is_client: bool = True):
        pass

    boolean_comparators = ["__gt__", "__ge__", "__lt__", "__le__"]

    ### Public API: framework-specific factory methods ###
    @classmethod
    @abstractmethod
    def create_shape(cls, shape_dims):
        """Factory method for creating a generic FrameworkShape."""
        pass

    @classmethod
    @abstractmethod
    def create_zeros(cls, shape, dtype, **kwargs):
        """Factory method for creating a generic zero FrameworkTensor."""
        pass

    @classmethod
    def create_wrapper(cls, wrapper_type, *args, **kwargs):
        """Factory method for creating a generic wrapper of type wrapper_type."""
        if wrapper_type is None:
            wrapper_type = syft.framework.Tensor

        return wrapper_type(*args, **kwargs)

    @classmethod
    def _transfer_methods_to_framework_class(
        hook_cls, framework_cls: type, from_cls: type, exclude: List[str]
    ):
        """Adds methods from the from_cls class to the framework_cls class.

        The class from_cls is a proxy class useful to avoid extending
        the native framework class directly.

        Args:
            framework_cls: The class to which we are adding methods, e.g.
                torch.Tensor or tf.Variable.
            from_cls: The class from which we are adding methods, e.g.
                TorchTensor, or TensorFlowVariable.
            exclude: A list of method names to exclude from the hooking process.
        """
        # For all methods defined in syft_type which are not internal methods
        # (like __class__, etc)
        for attr in dir(from_cls):
            if attr not in exclude:
                if hasattr(framework_cls, attr):
                    setattr(framework_cls, f"native_{attr}", getattr(framework_cls, attr))
                # Add to the native tensor this method
                setattr(framework_cls, attr, getattr(from_cls, attr))

    @classmethod
    def _perform_function_overloading(cls, parent_module_name, parent_module, func_name):

        # Where the overloading happens
        # 1. Get native function
        native_func = getattr(parent_module, func_name)
        # 2. Check it is a proper function
        if type(native_func) in [types.FunctionType, types.BuiltinFunctionType]:
            # 3. Build the hooked function
            new_func = cls._get_hooked_func(parent_module_name, func_name, native_func)
            # 4. Move the native function
            setattr(parent_module, f"native_{func_name}", native_func)
            # 5. Put instead the hooked one
            setattr(parent_module, func_name, new_func)

    @classmethod
    def _get_hooked_syft_method(cls, attr):
        """
        Hook a method in order to replace all args/kwargs syft/torch tensors with
        their child attribute, forward this method with the new args and new self,
        get response and "rebuild" the syft tensor wrapper upon all tensors found

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        @wraps(attr)
        def overloaded_syft_method(self, *args, **kwargs):
            """
            Operate the hooking
            """
            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                attr, self, args, kwargs
            )

            # Send it to the appropriate class and get the response
            response = getattr(new_self, attr)(*new_args, **new_kwargs)

            # For inplace methods, just directly return self
            if syft.framework.is_inplace_method(attr):
                return self

            # Put back SyftTensor on the tensors found in the response
            response = hook_args.hook_response(
                attr, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
            )

            return response

        return overloaded_syft_method

    @classmethod
    def _get_hooked_method(cls, tensor_type, method_name):
        """
        Hook a method in order to replace all args/kwargs syft/torch tensors with
        their child attribute if they exist
        If so, forward this method with the new args and new self, get response
        and "rebuild" the torch tensor wrapper upon all tensors found
        If not, just execute the native torch methodn

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        @wraps(getattr(tensor_type, method_name))
        def overloaded_native_method(self, *args, **kwargs):
            """
            Operate the hooking
            """

            if not hasattr(self, "child"):  # means that it's not a wrapper

                # if self is a natural tensor but the first argument isn't,
                # wrap self with the appropriate type and re-run
                if len(args) > 0 and hasattr(args[0], "child"):

                    # if we allow this for PointerTensors it opens the potential
                    # that we could accidentally serialize and send a tensor in the
                    # arguments
                    if not isinstance(args[0].child, PointerTensor):
                        self = type(args[0].child)().on(self, wrap=True)
                        args = [args[0]]
                        return overloaded_native_method(self, *args, **kwargs)

                method = getattr(self, f"native_{method_name}")
                # Run the native function with the new args

                try:
                    response = method(*args, **kwargs)

                except BaseException as e:
                    # we can make some errors more descriptive with this method
                    raise route_method_exception(e, self, args, kwargs)

            else:  # means that there is a wrapper to remove

                try:
                    # Replace all torch tensor with their child attribute
                    new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                        method_name, self, args, kwargs
                    )

                except BaseException as e:  # if there's a type mismatch, try to fix it!

                    try:
                        # if the first argument has no child (meaning it's probably raw data),
                        # try wrapping it with the type of self. We have to except PointerTensor
                        # because otherwise it can lead to inadvertently sending data to another
                        # machine
                        if not hasattr(args[0], "child") and not isinstance(
                            self.child, PointerTensor
                        ):
                            # TODO: add check to make sure this isn't getting around
                            # a security class

                            _args = []
                            _args.append(type(self)().on(args[0], wrap=False))
                            for a in args[1:]:
                                _args.append(a)

                            args = _args

                        # Replace all torch tensor with their child attribute
                        new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                            method_name, self, args, kwargs
                        )
                    except BaseException as e:
                        # we can make some errors more descriptive with this method
                        raise route_method_exception(e, self, args, kwargs)

                # Send the new command to the appropriate class and get the response
                method = getattr(new_self, method_name)
                response = method(*new_args, **new_kwargs)

                # For inplace methods, just directly return self
                if syft.framework.is_inplace_method(method_name):
                    return self

                # Put back the wrappers where needed
                response = hook_args.hook_response(
                    method_name,
                    response,
                    wrap_type=type(self),
                    new_self=self,
                    wrap_args=self.get_class_attributes(),
                )

            return response

        return overloaded_native_method

    @classmethod
    def _get_hooked_private_method(cls, method_name):
        """
        Hook a method in order to replace all args/kwargs syft/torch tensors with
        their child attribute if they exist
        If so, forward this method with the new args and new self, get response
        and "rebuild" the torch tensor wrapper upon all tensors found
        If not, just execute the native torch methodn

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        @wraps(method_name)
        def overloaded_native_method(self, *args, **kwargs):
            """
            Operate the hooking
            """
            if not hasattr(self, "child"):  # means that it's not a wrapper
                method = getattr(self, f"native_{method_name}")
                # Run the native function with the new args

                try:
                    response = method(*args, **kwargs)
                except BaseException as e:
                    # we can make some errors more descriptive with this method
                    raise route_method_exception(e, self, args, kwargs)

            else:  # means that there is a wrapper to remove
                try:
                    # Replace all torch tensor with their child attribute
                    new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                        method_name, self, args, kwargs
                    )
                except BaseException as e:
                    # we can make some errors more descriptive with this method
                    raise route_method_exception(e, self, args, kwargs)

                # Send the new command to the appropriate class and get the response
                method = getattr(new_self, method_name)
                response = method(*new_args, **new_kwargs)

                response.parents = (self.id, new_self.id)

                # For inplace methods, just directly return self
                if syft.framework.is_inplace_method(method_name):
                    return self

                # Put back the wrappers where needed
                response = hook_args.hook_response(
                    method_name,
                    response,
                    wrap_type=type(self),
                    new_self=self,
                    wrap_args=self.get_class_attributes(),
                )
                if args:
                    response.parents = (self, args[0])
                else:
                    response.parents = self
                response.command = method_name
            return response

        return overloaded_native_method

    @classmethod
    def _get_hooked_func(cls, public_module_name, func_api_name, func):
        """
        Hook a function in order to inspect its args and search for pointer
        or other syft tensors.
        - Calls to this function with normal tensors or numbers / string trigger
          usual behaviour
        - Calls with pointers send the command to the location of the pointer(s)
        - Calls with syft tensor will in the future trigger specific behaviour

        Args:
            public_module_name (str): the name of the public module you are
                hooking this function on (ie the same name that the user would import).
            attr (str): the method to hook
        Return:
            the hooked method
        """

        cmd_name = f"{public_module_name}.{func_api_name}"

        @wraps(func)
        def overloaded_func(*args, **kwargs):
            """
            Operate the hooking
            """

            try:
                tensor_type = (
                    type(args[0]) if not isinstance(args[0], (tuple, list)) else type(args[0][0])
                )
            except IndexError:
                tensor_type = syft.framework.Tensor

            command = (cmd_name, None, args, kwargs)

            try:
                handle_func_command = tensor_type.handle_func_command
            except AttributeError:
                handle_func_command = syft.framework.Tensor.handle_func_command

            response = handle_func_command(command)

            return response

        return overloaded_func
