from contextlib import contextmanager
import functools

import syft


class Trace:
    """
    Manage logging operations on the fly as they run

    Attributes:
        active: if True, recording of operation is active
        out_of_operation: it True, is ready to record the next operation,
            means that it's not inside the execution of an already recorded
            operation
        logs: the list of operations recorded
    """

    def __init__(self):
        self.active = False
        self.out_of_operation = True
        self.logs = []

    def clear(self):
        self.active = False
        self.out_of_operation = True
        self.logs = []

    @contextmanager
    def enabled(self):
        self.logs = []
        self.active = True
        self.out_of_operation = True
        try:
            yield self
        finally:
            self.active = False
            self.out_of_operation = False


def tracer(func_name=None, method_name=None):
    """
    This is a decorator which allows to record operations and their results
    when a function or a method is hooked

    This decorator is applied on overloaded_native_method and overloaded_func
    in the generic hook
    """

    def decorator(func):
        @functools.wraps(func)
        def trace_wrapper(*args, **kwargs):
            """
            The trace wrapper use two variables:

                syft.hook.trace.active: True if we are in the recording mode
                    of operations
                syft.hook.trace.out_of_operation: by default set to True, turns
                    to False when executing a recorded operation to prevent from
                    recording sub operations
            """

            if syft.hook.trace.active and syft.hook.trace.out_of_operation:
                # Select if the tracer records a function or a method, not none or both
                assert (func_name is None) ^ (method_name is None)

                cmd_name = func_name or method_name

                if method_name is not None:
                    # We extract the self with args[0]
                    command = (cmd_name, args[0], args[1:], kwargs)
                else:
                    command = (cmd_name, None, args, kwargs)

                syft.hook.trace.out_of_operation = False

                response = func(*args, **kwargs)

                syft.hook.trace.out_of_operation = True

                syft.hook.trace.logs.append((command, response))
            else:
                response = func(*args, **kwargs)

            return response

        return trace_wrapper

    return decorator
