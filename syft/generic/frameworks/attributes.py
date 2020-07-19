from abc import ABC
from abc import abstractmethod
from types import ModuleType
from typing import Union
from typing import Callable
from typing import Any

from syft.generic.frameworks.hook.hook import FrameworkHook

allowed_commands = set()


class FrameworkAttributes(ABC):
    _allowed_commands = None

    @abstractmethod
    def __init__(self, framework: ModuleType, hook: FrameworkHook):
        pass

    @property
    def allowed_commands(self):
        if self._allowed_commands is None:
            self._allowed_commands = allowed_commands
        return self._allowed_commands

    @allowed_commands.setter
    def allowed_commands(self, new_commands):
        self._allowed_commands = new_commands

    # Forcing subclasses to define a class-level constant; see
    # https://stackoverflow.com/a/53417582 for nuance
    @property
    @classmethod
    @abstractmethod
    def ALIAS(cls):
        pass

    @property
    @classmethod
    @abstractmethod
    def Tensor(cls):
        """Default Tensor wrapper."""
        pass

    @abstractmethod
    def is_inplace_method(self, method_name):
        """Determine if a method is inplace or not.

        Framework-dependent, see subclasses for details.

        Args:
            method_name: The name for the method.
        Returns:
            Boolean denoting if the method is inplace or not.
        """
        pass

    @abstractmethod
    def is_global_state_change_method(self, method_name):
        """Determine if a method updates global module state.

        Framework-dependent, see subclasses for details.

        Args:
            method_name: The name for the method.
        Returns:
            Boolean denoting if the method updates global module state.
        """
        pass

    def _command_guard(
        self, command: str, get_native: bool = False
    ) -> Union[Callable[..., Any], str]:
        """Check command can be safely used.

        Args:
            command: A string indicating command name.
            get_native: A boolean parameter (default False) to indicate whether
                to return the command name or the native torch function. If
                False, return command name else return the native torch
                function.

        Returns:
            The command name or a native framework function
        """
        if command not in self.allowed_commands:
            raise RuntimeError(f'Command "{command}" is not a supported {self.ALIAS} operation.')
        if get_native:
            return self.native_commands[command]
        return command

    def _is_command_valid_guard(self, command: str) -> bool:
        """Validate the command.

        Indicates whether a command is valid with respect to the framework
        guard.

        Args:
            command: A string indicating command to test.
            framework_domain: A string indicating the framework domain or
                module in which the command is supposed to be, e.g.
                dir(torch), dir(torch.Tensor), dir(tensorflow), etc. (roughly)

        Returns:
            A boolean indicating whether the command is valid.
        """
        try:
            self._command_guard(command)
        except RuntimeError:
            return False
        return True

    @classmethod
    def get_native_framework_name(cls, attr: str) -> str:
        """Return the name of the native command for the given hooked command.

        Args:
            attr: A string indicating the hooked command name (ex: torch.add)

        Returns:
            The name of the native command (ex: torch.native_add)
        """
        parts = attr.split(".")
        parts[-1] = "native_" + parts[-1]
        native_func_name = ".".join(parts)
        return native_func_name
