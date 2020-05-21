from abc import ABC
from abc import abstractmethod

import inspect
import re
import types

import syft


class TensorHook(ABC):
    """Hook for ALL THE TENSOR THINGS that must be overloaded and/or modified"""

    @abstractmethod
    def _hook_native_tensor(self, tensor_type: type, syft_type: type):
        """Add PySyft-specific tensor functionality to the given tensor type.

        See framework-specific implementations for more details.
        """
        # _hook_native_tensor is framework-specific, but it calls the methods
        # defined below!
        pass

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
