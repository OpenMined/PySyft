from contextlib import contextmanager
import functools

import syft


class Trace:
    """
    Manage logging actions on the fly as they run

    Attributes:
        active: if True, recording of action is active
        out_of_action: it True, is ready to record the next action,
            means that it's not inside the execution of an already recorded
            action
        logs: the list of actions recorded
    """

    def __init__(self):
        self.active = False
        self.out_of_action = True
        self.logs = []

    def clear(self):
        self.active = False
        self.out_of_action = True
        self.logs = []

    @contextmanager
    def enabled(self):
        self.logs = []
        self.active = True
        self.out_of_action = True
        try:
            yield self
        finally:
            self.active = False
            self.out_of_action = False


def tracer(func_name=None, method_name=None):
    """
    This is a decorator which allows to record actions and their results
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
                    of actions
                syft.hook.trace.out_of_action: by default set to True, turns
                    to False when executing a recorded action to prevent from
                    recording sub actions
            """

            if not hasattr(syft, "hook") or syft.hook == None:
                return func(*args, **kwargs)

            if syft.hook.trace.active and syft.hook.trace.out_of_action:
                # Select if the tracer records a function or a method, not none or both
                assert (func_name is None) ^ (method_name is None)

                cmd_name = func_name or method_name

                if method_name is not None:
                    # We extract the self with args[0]
                    command = (cmd_name, args[0], args[1:], kwargs)
                else:
                    command = (cmd_name, None, args, kwargs)

                syft.hook.trace.out_of_action = False

                response = func(*args, **kwargs)

                syft.hook.trace.out_of_action = True

                syft.hook.trace.logs.append((command, response))
            else:
                response = func(*args, **kwargs)

            return response

        return trace_wrapper

    return decorator
