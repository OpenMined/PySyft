from typing import Union
from typing import Callable
from typing import Any
from types import ModuleType


class TorchAttributes(object):
    """
    TorchAttributes is a special class where all custom attributes related
    to the torch module can be added. Any global parameter, configuration,
    or reference relating to PyTorch should be stored here instead of
    attaching it directly to some other part of the global namespace.

    The main reason we need this is because the hooking process occasionally
    needs to save global objects, notably including what methods to hook and
    what methods to NOT hook.
    """

    def __init__(self, torch: ModuleType) -> None:
        """
        Initialization of the TorchAttributes class. This will hold all necessary
        attributes PySyft needs.

        Args:
            torch: torch module
        """

        # SECTION: List all functions in torch module that we want to overload

        # List modules that we will hook
        self.torch_modules = {"torch": torch, "torch.nn.functional": torch.nn.functional}

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
        self.exclude = ["save", "load", "typename", "is_tensor", "manual_seed"]

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

    def _command_guard(
        self, command: str, torch_domain: str, get_native: bool = False
    ) -> Union[Callable[..., Any], str]:
        """
        Check that a command is in a given torch_domain and can be safely used

        Args:
            command (str): the command name
            torch_domain (str): name of the torch domain or module in which the command is
                supposed to be
            get_native (boolean): if False (default), return the command name. If True, return
                the native command function

        Returns:
            The command name or a native torch function
        """
        if command not in self.allowed_commands[torch_domain]:
            raise RuntimeError(f'Command "{command}" is not a supported Torch operation.')
        if get_native:
            return self.native_commands[torch_domain][command]
        return command

    def _is_command_valid_guard(self, command: str, torch_domain: str) -> bool:
        """
        Indicates whether a command is valid with respect to the torch guard

        Args:
            command (str): the command to test
            torch_domain (str): the torch domain or module in which the command is supposed to be

        Returns:
            A boolean
        """
        try:
            self._command_guard(command, torch_domain)
        except RuntimeError:
            return False
        return True

    def eval_torch_modules(self) -> None:
        """
        For each torch command functions in native_commands, transform the dictionary so
        that to each key, which is the name of the hooked command, now corresponds a value
        which is the evaluated native name of the command, namely the native command.
        Note that we don't do this for methods.
        """
        for cmd_name, native_cmd_name in self.native_commands["torch_modules"].items():
            if cmd_name not in self.torch_exclude:
                self.native_commands["torch_modules"][cmd_name] = self.eval_torch_modules_functions[
                    cmd_name
                ]

    @staticmethod
    def get_native_torch_name(attr: str) -> str:
        """
        Return the name of the native command given the name of the hooked command.

        Args:
            attr (str): the name of the hooked command (ex: torch.add)

        Returns:
            the name of the native command (ex: torch.native_add)
        """
        parts = attr.split(".")
        parts[-1] = "native_" + parts[-1]
        native_func_name = ".".join(parts)
        return native_func_name
