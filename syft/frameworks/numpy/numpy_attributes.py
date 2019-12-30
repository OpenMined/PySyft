from types import ModuleType
from typing import Union
from typing import Callable
from typing import Any

from syft.generic.frameworks.attributes import FrameworkAttributes
from syft import ndarray as SyftNdarray
from syft.frameworks.numpy.tensors.interpreters.native import NumpyTensor


class NumpyAttributes(FrameworkAttributes):
    """Adds torch module related custom attributes.

    TorchAttributes is a special class where all custom attributes related
    to the torch module can be added. Any global parameter, configuration,
    or reference relating to PyTorch should be stored here instead of
    attaching it directly to some other part of the global namespace.

    The main reason we need this is because the hooking process occasionally
    needs to save global objects, notably including what methods to hook and
    what methods to NOT hook.

    This will hold all necessary attributes PySyft needs.

    Args:
        numpy: A ModuleType indicating the numpy module
        hook: A NumpyHook to stash
    """

    ### Subclasses must provide the following class attributes ###
    ALIAS = "numpy"
    Tensor = NumpyTensor

    def __init__(self, numpy: ModuleType, hook: ModuleType) -> None:
        """Initialization of the NumpyAttributes class."""

        # Stash the hook here for global access elsewhere
        self.hook = hook

        # SECTION: List all functions in torch module that we want to overload

        # List modules that we will hook
        self.numpy_modules = {
            "numpy": numpy,
        }

        # Set of all function names with module as prefix in the modules to hook
        self._numpy_functions = {
            f"{module_name}.{func_name}"
            for module_name, numpy_module in self.numpy_modules.items()
            for func_name in dir(numpy_module)
        }

        # Add special functions to exclude from the hook **in alphabetical order**
        # Reasons can be:
        # - Used for internal process like printing tensors
        # - Don't use tensors so are bound to have local executions
        # - etc
        # DON'T put here:
        # - functions like native_*
        # - functions that could use pointers or syft tensors
        self.exclude = [
            "arange",
            "as_tensor",
            "from_numpy",
            "get_default_dtype",
            "get_device",
            "get_file_path",
            "get_num_threads",
            "get_rng_state",
            "has_names",
            "int",
            "int16",
            "int32",
            "int64",
            "int8",
            "is_anomaly_enabled",
            "is_complex",
            "is_distributed",
            "is_floating_point",
            "is_grad_enabled",
            "is_nonzero",
            "is_same_size",
            "is_signed",
            "is_storage",
            "is_tensor",
            "isclose",
            "isfinite",
            "load",
            "manual_seed",
            "ones",
            "range",
            "save",
            "set_",
            "set_num_threads",
            "short",
            "size",
            "storage",
            "storage_offset",
            "stride",
            "tensor",
            "typename",
            "zeros",
        ]

        self.worker_methods = ["tensor", "rand", "zeros", "randn", "randint"]

        # SECTION: Build the guard, that define which functions or methods can be safely called by
        # external or local workers

        # Add all tensor types
        self.guard = {
            "Tensor": SyftNdarray,
            #"DoubleTensor": torch.DoubleTensor,
            #"HalfTensor": torch.HalfTensor,
            #"ByteTensor": torch.ByteTensor,
            #"CharTensor": torch.CharTensor,
            #"ShortTensor": torch.ShortTensor,
            #"IntTensor": torch.IntTensor,
            #"LongTensor": torch.LongTensor,
            #"Parameter": torch.nn.Parameter,
        }

        # Allow the `syft.` prefix to be used
        keys = list(self.guard.keys())
        for key in keys:
            self.guard[f"syft.{key}"] = self.guard[key]

        # Concatenate torch functions
        self.allowed_commands = self._numpy_functions

        # The equivalent concatenation of native torch function names and native torch method names
        self.native_commands = {
            command_name: self.get_native_framework_name(command_name)
            for command_name in self.allowed_commands
        }

        self.command_guard = self._command_guard

        # Dict {method_name: <is_inplace:bool>
        self.inplace_methods = {}

    def is_inplace_method(self, method_name):
        """Determine if a method is inplace or not.

        Check if the method ends by _ and is not a __xx__, then stash for
        constant-time lookup.

        Args:
            method_name: The name for the method.
        Returns:
            Boolean denoting if the method is inplace or not.
        """
        try:
            return self.inplace_methods[method_name]
        except KeyError:
            is_inplace = method_name[-1] == "_" and "__" not in method_name
            self.inplace_methods[method_name] = is_inplace
            return is_inplace
