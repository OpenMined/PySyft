from types import ModuleType

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.generic.frameworks.types import FrameworkTensor


class FrameworkWrapper:
    def __init__(self, package, role):
        self.package = package
        self.role = role

    def __getattr__(self, attr_name):
        package_attr = getattr(self.package, attr_name)
        # Forward directly the attribute if it's not a function
        if not callable(package_attr):
            # If it's a sub-module, wrap that for tracing too
            if isinstance(package_attr, ModuleType):
                return FrameworkWrapper(package_attr, self.role)
            else:
                return package_attr

        def trace_wrapper(*args, **kwargs):
            """creates placeholders and registers ComputationAction to role"""
            cmd_name = ".".join((self.package.__name__, attr_name))
            command = (cmd_name, None, args, kwargs)

            result = package_attr(*args, **kwargs)

            if isinstance(result, PlaceHolder) or (
                isinstance(result, (list, tuple))
                and any(isinstance(r, PlaceHolder) for r in result)
            ):
                # In this case, the tracing was already done in Placeholder.handle_func_command
                return result

            if isinstance(result, FrameworkTensor):
                result = PlaceHolder.create_from(result, role=self.role, tracing=True)
                self.role.register_action(
                    (command, result), sy.execution.computation.ComputationAction
                )
            elif isinstance(result, (list, tuple)):
                result = tuple(
                    PlaceHolder.create_from(r, role=self.role, tracing=True) for r in result
                )
                self.role.register_action(
                    (command, result), sy.execution.computation.ComputationAction
                )
            else:
                self.role.register_action(
                    (command, None), sy.execution.computation.ComputationAction
                )

            return result

        return trace_wrapper
