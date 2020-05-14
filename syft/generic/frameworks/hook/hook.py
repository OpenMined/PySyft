from abc import ABC
from abc import abstractmethod
from functools import wraps
import inspect
import re
import types
from typing import List, Tuple

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.string import String
from syft.generic.pointers.string_pointer import StringPointer
from syft.generic.object import _apply_args
from syft.workers.base import BaseWorker

from syft.exceptions import route_method_exception
from syft.exceptions import TensorsNotCollocatedException


class FrameworkHook(ABC):
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
                new_method = self._get_hooked_method(tensor_type, attr)
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

        to_overload = self.boolean_comparators.copy()

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

    def _hook_syft_placeholder_methods(self, tensor_type: type, syft_type: type):
        """
        Slight variant of _hook_syft_tensor_methods, which adds the boolean
        comparators to the hooking
        """

        def create_tracing_method(base_method, name):
            def tracing_method(self, *args, **kwargs):
                response = base_method(self, *args, **kwargs)
                command = (name, self, args, kwargs), response
                if self.tracing:
                    self.role.register_action(command, syft.execution.computation.ComputationAction)
                return response

            return tracing_method

        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(syft_type) or attr in self.boolean_comparators:
                new_method = create_tracing_method(self._get_hooked_syft_method(attr), attr)
                setattr(syft_type, attr, new_method)

    def _hook_private_tensor_methods(self, tensor_type: type, syft_type: type):
        """
        Add hooked version of all methods of the tensor_type to the
        Private Tensor: It'll add references to its parents and save
        command/actions history.
        """
        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(syft_type):
                new_method = self._get_hooked_private_method(attr)
                setattr(syft_type, attr, new_method)

    def _hook_pointer_tensor_methods(self, tensor_type):
        """
        Add hooked version of all methods of the tensor_type to the
        Pointer tensor: instead of performing the native tensor
        method, it will be sent remotely to the location the pointer
        is pointing at.
        """

        # Use a pre-defined list to select the methods to overload
        for attr in self.to_auto_overload[tensor_type]:
            if attr not in dir(PointerTensor) or attr in self.boolean_comparators:
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

    def _hook_string_methods(self, owner):

        # Set the default owner
        setattr(String, "owner", owner)

        for attr in dir(str):

            if attr in String.methods_to_hook:

                # Create the hooked method
                new_method = self._get_hooked_string_method(attr)

                # Add the hooked method
                setattr(String, attr, new_method)

    def _hook_string_pointer_methods(self):

        for attr in dir(String):

            if attr in String.methods_to_hook:

                # Create the hooked method
                new_method = self._get_hooked_string_pointer_method(attr)

                # Add the hooked method
                setattr(StringPointer, attr, new_method)

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
                            # TODO: add check to make sure this isn't getting around a security class

                            _args = list()
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
            response = owner.send_command(location, attr, self, args, kwargs)

            # For inplace methods, just directly return self
            if syft.framework.is_inplace_method(attr):
                return self

            return response

        return overloaded_pointer_method

    @classmethod
    def _get_hooked_multi_pointer_method(cls, attr):
        """
        Hook a method to send it multiple remote workers

        Args:
            attr (str): the method to hook
        Return:
            the hooked method
        """

        def dispatch(args_, k):
            return map(lambda x: x[k] if isinstance(x, dict) else x, args_)

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

    @classmethod
    def _string_input_args_adaptor(cls, args_: Tuple[object]):
        """
           This method is used when hooking String methods.

           Some 'String' methods which are overriden from 'str'
           such as the magic '__add__' method
           expects an object of type 'str' as its first
           argument. However, since the '__add__' method
           here is hooked to a String type, it will receive
           arguments of type 'String' not 'str' in some cases.
           This won't worker for the underlying hooked method
           '__add__' of the 'str' type.
           That is why the 'String' argument to '__add__' should
           be peeled down to 'str'

           Args:
               args_: A tuple or positional arguments of the method
                     being hooked to the String class.

           Return:
               A list of adapted positional arguments.

        """

        new_args = []

        for arg in args_:

            # If 'arg' is an object of type String
            # replace it by and 'str' object
            if isinstance(arg, String):
                new_args.append(arg.child)
            else:
                new_args.append(arg)

        return new_args

    @classmethod
    def _wrap_str_return_value(cls, _self, attr: str, value: object):

        # The outputs of the following attributed won't
        # be wrapped
        ignored_attr = set(["__str__", "__repr__", "__format__"])

        if isinstance(value, str) and attr not in ignored_attr:

            return String(object=value, owner=_self.owner)

        return value

    @classmethod
    def _get_hooked_string_method(cls, attr):
        """
           Hook a `str` method to a corresponding method  of
          `String` with the same name.

           Args:
               attr (str): the method to hook
           Return:
               the hooked method

        """

        @wraps(attr)
        def overloaded_attr(_self, *args, **kwargs):

            args = cls._string_input_args_adaptor(args)

            # Call the method of the core builtin type
            native_response = getattr(_self.child, attr)(*args, **kwargs)

            # Some return types should be wrapped using the String
            # class. For instance, if 'foo' is an object of type
            # 'String' which wraps 'str'. calling foo.upper()
            # should also be of type 'String' not 'str'.
            # However, the return value of foo.__str__ should
            # be of type 'str'.
            response = cls._wrap_str_return_value(_self, attr, native_response)

            return response

        return overloaded_attr

    @classmethod
    def _get_hooked_string_pointer_method(cls, attr):
        """
           Hook a `String` method to a corresponding method  of
          `StringPointer` with the same name.

           Args:
               attr (str): the method to hook
           Return:
               the hooked method

        """

        @wraps(attr)
        def overloaded_attr(_self, *args, **kwargs):
            """
            Operate the hooking
            """

            owner = _self.owner
            location = _self.location
            # id_at_location = self.id_at_location

            # Create a 'command' variable  that is understood by
            # the send_command() method of a worker.
            # command = (attr, id_at_location, args, kwargs)

            # send the command
            response = owner.send_command(location, attr, _self, args, kwargs)

            return response

        return overloaded_attr
