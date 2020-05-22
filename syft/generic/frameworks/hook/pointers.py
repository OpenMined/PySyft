from abc import ABC
from functools import wraps

import syft

from syft.exceptions import TensorsNotCollocatedException
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.pointer_tensor import PointerTensor


class PointerHook(ABC):
    """Hook for ALL THE POINTER THINGS that must be overloaded and/or modified"""

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
