from typing import Union
from typing import Callable
from typing import Any
from types import ModuleType


class TorchAttributes(object):
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
        torch: A ModuleType indicating the torch module
        hook: A ModuleType indicating the modules to hook
    """

    def __init__(self, torch: ModuleType, hook: ModuleType) -> None:
        """Initialization of the TorchAttributes class."""

        # SECTION: List all functions in torch module that we want to overload

        self.hook = hook

        # List modules that we will hook
        self.torch_modules = {
            "torch": torch,
            "torch.functional": torch.functional,
            "torch.nn.functional": torch.nn.functional,
        }

        # List all the function names with module as prefix in the modules to hook
        self.torch_modules_functions = {
            f"{module_name}.{func_name}"
            for module_name, torch_module in self.torch_modules.items()
            for func_name in dir(torch_module)
        }

        # Store reference to all torch functions by string name stored in torch_modules_functions
        self.eval_torch_modules_functions = {
            f"{module_name}.{func_name}": getattr(torch_module, func_name)
            for module_name, torch_module in self.torch_modules.items()
            for func_name in dir(torch_module)
        }

        # Add special functions to exclude from the hook
        self.exclude = [
            "arange",
            "save",
            "load",
            "typename",
            "is_tensor",
            "manual_seed",
            "storage",
            "storage_offset",
            "size",
            "stride",
            "from_numpy",
            "set_",
            "get_default_dtype",
            "get_device",
            "get_file_path",
            "get_num_threads",
            "get_rng_state",
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
            "isfinite" "load",
            "long",
            "native_add",
            "native_batch_norm",
            "native_clone",
            "native_norm",
            "native_pow",
            "native_resize_as_",
            "native_tensor",
            "native_zero_",
            "ones",
            "rand",
            "randint",
            "randn_like",
            "range",
            "save",
            "short",
            "zeros",
            "tensor",
            "get_default_dtype",
            "is_grad_enabled",
            "is_nonzero",
            "is_storage",
            "is_tensor",
            "isfinite",
            "load",
            "randperm",
        ]

        # SECTION: List all torch tensor methods we want to overload
        self.tensor_types = [torch.Tensor]

        self.tensorvar_methods = list(
            {method for tensorvar in self.tensor_types for method in dir(tensorvar)}
        )
        self.tensorvar_methods += ["get_shape", "share", "fix_precision", "decode", "end_get"]

        # SECTION: Build the guard, that define which functions or methods can be safely called by
        # external or local workers

        # Add all tensor types
        self.guard = {
            "FloatTensor": torch.FloatTensor,
            "DoubleTensor": torch.DoubleTensor,
            "HalfTensor": torch.HalfTensor,
            "ByteTensor": torch.ByteTensor,
            "CharTensor": torch.CharTensor,
            "ShortTensor": torch.ShortTensor,
            "IntTensor": torch.IntTensor,
            "LongTensor": torch.LongTensor,
            "Parameter": torch.nn.Parameter,
        }

        # Allow the `syft.` prefix to be used
        keys = list(self.guard.keys())
        for key in keys:
            self.guard[f"syft.{key}"] = self.guard[key]

        # Concatenate torch functions and torch methods
        self.allowed_commands = {
            "tensorvar_methods": self.tensorvar_methods,
            "torch_modules": self.torch_modules_functions,
        }

        # The equivalent concatenation of native torch function names and native torch method names
        self.native_commands = {
            command_type: {cmd: self.get_native_torch_name(cmd) for cmd in commands}
            for command_type, commands in self.allowed_commands.items()
        }

        self.command_guard = self._command_guard

        # Dict {method_name: <is_inplace:bool>
        self.inplace_methods = {}

    def _command_guard(
        self, command: str, torch_domain: str, get_native: bool = False
    ) -> Union[Callable[..., Any], str]:
        """Checks command is in a given torch_domain and can be safely used.

        Args:
            command: A string indicating command name.
            torch_domain: A string indicating torch domain name or module in
                which the command is supposed to be.
            get_native: A boolean parameter (default False) to indicate whether
                to return the command name or the native torch function. If
                False, return command name else return the native torch
                function.

        Returns:
            The command name or a native torch function
        """
        if command not in self.allowed_commands[torch_domain]:
            raise RuntimeError(f'Command "{command}" is not a supported Torch operation.')
        if get_native:
            return self.native_commands[torch_domain][command]
        return command

    def _is_command_valid_guard(self, command: str, torch_domain: str) -> bool:
        """Validates the command.

        Indicates whether a command is valid with respect to the torch guard

        Args:
            command: A string indicating command to test.
            torch_domain: A string indicating the torch domain or module in
                which the command is supposed to be.

        Returns:
            A boolean indicating whether the command is valid.
        """
        try:
            self._command_guard(command, torch_domain)
        except RuntimeError:
            return False
        return True

    def eval_torch_modules(self) -> None:
        """Builds a mapping between the hooked and native commands.

        For each torch command functions in native_commands, transform the
        dictionary so that to each key, which is the name of the hooked
        command, now corresponds a value which is the evaluated native name of
        the command, namely the native command.

        Note that we don't do this for methods.
        """
        for cmd_name, native_cmd_name in self.native_commands["torch_modules"].items():
            if cmd_name not in self.torch_exclude:
                self.native_commands["torch_modules"][cmd_name] = self.eval_torch_modules_functions[
                    cmd_name
                ]

    @staticmethod
    def get_native_torch_name(attr: str) -> str:
        """Returns the name of the native command for the given hooked command.

        Args:
            attr: A string indicating the hooked command name (ex: torch.add)

        Returns:
            The name of the native command (ex: torch.native_add)
        """
        parts = attr.split(".")
        parts[-1] = "native_" + parts[-1]
        native_func_name = ".".join(parts)
        return native_func_name

    def is_inplace_method(self, method_name):
        """
        Says if a method is inplace or not by test if it ends by _ and is not a __xx__
        :param method_name: the name for the method
        :return: boolean
        """
        try:
            return self.inplace_methods[method_name]
        except KeyError:
            is_inplace = method_name[-1] == "_" and "__" not in method_name
            self.inplace_methods[method_name] = is_inplace
            return is_inplace

    @staticmethod
    def apply_fix16922(torch):
        """
        Apply the fix made in PR16922 of PyTorch until people use PyTorch 1.0.2
        :param torch: the pytorch module
        """
        broken_funcs = [
            "max_pool1d",
            "max_pool2d",
            "max_pool3d",
            "adaptive_max_pool1d",
            "adaptive_max_pool2d",
            "adaptive_max_pool3d",
        ]
        for broken_func in broken_funcs:
            getattr(torch.nn.functional, broken_func).__module__ = "torch.nn.functional"
            getattr(torch.nn.functional, broken_func).__name__ = broken_func
