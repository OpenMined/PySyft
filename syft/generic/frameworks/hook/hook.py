from abc import ABC
from abc import abstractmethod
from functools import wraps
import inspect
import re
import types
from typing import List

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.object import initialize_object
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.tensor import _apply_args
from syft.workers.base import BaseWorker

from syft.exceptions import route_method_exception
from syft.exceptions import TensorsNotCollocatedException


class FrameworkHook(ABC):
    @abstractmethod
    def __init__(self, framework_module, local_worker: BaseWorker = None, is_client: bool = True):
        pass

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

    ### Standardized, framework-specific methods ###
    @abstractmethod
    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Add PySyft-specific tensor functionality to the given tensor type.

        See framework-specific implementations for more details.
        """
        # _hook_native_tensor is framework-specific, but it calls the methods
        # defined below!
        pass

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

    ### Generics methods ###
    def _hook_native_methods(self, tensor_type: type):
        """
        Add hooked version of all methods of to_auto_overload[tensor_type]
        to the tensor_type; instead of performing the native tensor
        method, the hooked version will be called

        Args:
            tensor_type: the tensor_type which holds the methods
        """
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            # if we haven't already overloaded this function
            if f"native_{attr}" not in dir(tensor_type):
                native_method = getattr(tensor_type, attr)
                setattr(tensor_type, f"native_{attr}", native_method)
                new_method = self._get_hooked_method(attr)
                setattr(tensor_type, attr, new_method)

    def _hook_properties(hook_self, tensor_type: type):
        """Overloads tensor_type properties.

        If you're not sure how properties work, read:
        https://www.programiz.com/python-programming/property
        Args:
            tensor_type: The tensor class which is having properties
                added to it.
        """

        @property
        def location(self):
            if hasattr(self, "child"):
                return self.child.location
            else:
                return None

        tensor_type.location = location

        @property
        def id_at_location(self):
            return self.child.id_at_location

        tensor_type.id_at_location = id_at_location

        @property
        def id(self):
            if not hasattr(self, "_syft_id"):
                self._syft_id = syft.ID_PROVIDER.pop()
            return self._syft_id

        @id.setter
        def id(self, new_syft_id):
            self._syft_id = new_syft_id
            return self

        tensor_type.id = id

        @property
        def owner(self):
            if not hasattr(self, "_owner"):
                self._owner = hook_self.local_worker
            return self._owner

        @owner.setter
        def owner(self, new_owner):
            self._owner = new_owner
            return self

        tensor_type.owner = owner

        @property
        def is_wrapper(self):
            if not hasattr(self, "_is_wrapper"):
                self._is_wrapper = False
            return self._is_wrapper

        @is_wrapper.setter
        def is_wrapper(self, it_is_a_wrapper):
            self._is_wrapper = it_is_a_wrapper
            return self

        tensor_type.is_wrapper = is_wrapper

        def dim(self):
            return len(self.shape)

        tensor_type.dim = dim

    def _which_methods_should_we_auto_overload(self, tensor_type: type):
        """Creates a list of Torch methods to auto overload.

        By default, it looks for the intersection between the methods of
        tensor_type and torch_type minus those in the exception list
        (syft.torch.exclude).

        Args:
            tensor_type: Iterate through the properties of this tensor type.
            syft_type: Iterate through all attributes in this type.

        Returns:
            A list of methods to be overloaded.
        """

        boolean_comparators = ["__gt__", "__ge__", "__lt__", "__le__"]

        to_overload = boolean_comparators

        native_pattern = re.compile("native*")

        for attr in dir(tensor_type):

            # Conditions for not overloading the method
            # TODO[jvmancuso] separate func exclusion from method exclusion
            if attr in syft.framework.exclude:
                continue
            if not hasattr(tensor_type, attr):
                continue

            lit = getattr(tensor_type, attr)
            is_base = attr in dir(object)
            is_desc = inspect.ismethoddescriptor(lit)
            is_func = isinstance(lit, types.FunctionType)
            is_overloaded = native_pattern.match(attr) is not None

            if (is_desc or is_func) and not is_base and not is_overloaded:
                to_overload.append(attr)

        return set(to_overload)

    def _hook_syft_tensor_methods(self, tensor_type: type, syft_type: type):
        """
        Add hooked version of all methods of to_auto_overload[tensor_type]
        to the syft_type, so that they act like regular tensors in
        terms of functionality, but instead of performing the native tensor
        method, it will be forwarded to each share when it is relevant

        Args:
            tensor_type: The tensor type to which we are adding methods.
            syft_type: the syft_type which holds the methods
        """

        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(syft_type):
                new_method = self._get_hooked_syft_method(attr)
                setattr(syft_type, attr, new_method)

    def _hook_pointer_tensor_methods(self, tensor_type):
        """
        Add hooked version of all methods of the tensor_type to the
        Pointer tensor: instead of performing the native tensor
        method, it will be sent remotely to the location the pointer
        is pointing at.
        """

        boolean_comparators = ["__gt__", "__ge__", "__lt__", "__le__"]

        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(PointerTensor) or attr in boolean_comparators:
                new_method = self._get_hooked_pointer_method(attr)
                setattr(PointerTensor, attr, new_method)

    def _hook_object_pointer_methods(self, framework_cls):
        """
        Add hooked version of all methods of the framework_cls to the
        ObjectPointer: instead of performing the native object
        method, it will be sent remotely to the location the pointer
        is pointing at.
        """

        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[framework_cls]:
            new_method = self._get_hooked_pointer_method(attr)
            setattr(ObjectPointer, attr, new_method)

    def _hook_multi_pointer_tensor_methods(self, tensor_type):
        """
        Add hooked version of all methods of the torch Tensor to the
        Multi Pointer tensor: instead of performing the native tensor
        method, it will be sent remotely for each pointer to the
        location it is pointing at.
        """

        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(MultiPointerTensor):
                new_method = self._get_hooked_multi_pointer_method(attr)
                setattr(MultiPointerTensor, attr, new_method)

    def _add_registration_to___init__(hook_self, tensor_type: type, is_tensor: bool = False):
        """Adds several attributes to the tensor.

        Overload tensor_type.__init__ to add several attributes to the tensor
        as well as (optionally) registering the tensor automatically.
        TODO: auto-registration is disabled at the moment, this might be bad.

        Args:
            tensor_type: The class of the tensor being hooked
            torch_tensor: An optional boolean parameter (default False) to
                specify whether to skip running the native initialization
                logic. TODO: this flag might never get used.
        """
        if "native___init__" not in dir(tensor_type):
            tensor_type.native___init__ = tensor_type.__init__

        def new___init__(self, *args, owner=None, id=None, register=True, **kwargs):
            initialize_object(
                hook=hook_self,
                obj=self,
                id=id,
                reinitialize=not is_tensor,
                init_args=args,
                init_kwargs=kwargs,
            )

        tensor_type.__init__ = new___init__

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

            # Put back SyftTensor on the tensors found in the response
            response = hook_args.hook_response(
                attr, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
            )

            return response

        return overloaded_syft_method

    @classmethod
    def _get_hooked_method(cls, method_name):
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

                # For inplace methods, just directly return self
                if syft.framework.is_inplace_method(method_name):
                    return self

                # Put back the wrappers where needed
                response = hook_args.hook_response(
                    method_name, response, wrap_type=type(self), new_self=self
                )

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

    @classmethod
    def _get_hooked_pointer_method(cls, attr):
        """
        Hook a method to send it to remote worker

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        @wraps(attr)
        def overloaded_pointer_method(self, *args, **kwargs):
            """
            Operate the hooking
            """
            pointer = self
            # Get info on who needs to send where the command
            owner = pointer.owner
            location = pointer.location

            if len(args) > 0:
                if isinstance(args[0], ObjectPointer):
                    if args[0].location.id != location.id:
                        raise TensorsNotCollocatedException(pointer, args[0], attr)

            # Send the command
            command = (attr, self, args, kwargs)

            response = owner.send_command(location, command)

            # For inplace methods, just directly return self
            if syft.framework.is_inplace_method(attr):
                return self

            return response

        return overloaded_pointer_method

    @classmethod
    def _get_hooked_multi_pointer_method(cls, attr):
        """
        Hook a method to send it multiple recmote workers

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        def dispatch(args, k):
            return map(lambda x: x[k] if isinstance(x, dict) else x, args)

        @wraps(attr)
        def overloaded_attr(self, *args, **kwargs):
            """
            Operate the hooking
            """

            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                attr, self, args, kwargs
            )

            results = {}
            for k, v in new_self.items():
                results[k] = v.__getattribute__(attr)(*dispatch(new_args, k), **new_kwargs)

            # Put back MultiPointerTensor on the tensors found in the response
            response = hook_args.hook_response(
                attr, results, wrap_type=MultiPointerTensor, wrap_args=self.get_class_attributes()
            )

            return response

        return overloaded_attr
