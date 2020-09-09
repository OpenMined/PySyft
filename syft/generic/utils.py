from syft.generic.frameworks.attributes import allowed_commands
import syft as sy


class memorize(dict):
    """
    This is a decorator to cache a function output when the function is
    deterministic and the input space is small. In such condition, the
    function will be called many times to perform the same computation
    so we want this computation to be cached.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        return self[key]

    def __missing__(self, key):
        args, kwargs = key
        kwargs = {k: v for k, v in kwargs}
        result = self[key] = self.func(*args, **kwargs)
        return result


def allow_command(func):
    module = func.__module__
    func_name = f"{module}{'.' if module else ''}{func.__name__}"
    allowed_commands.update({func_name})
    return func


def remote(func, location):
    module = func.__module__
    command_name = f"{module}{'.' if module else ''}{func.__name__}"

    worker = sy.local_worker

    if isinstance(location, str):
        location = worker.get_worker(location)

    def remote_exec(*args, return_value=False, return_arity=1, **kwargs):

        response_ids = tuple(sy.ID_PROVIDER.pop() for _ in range(return_arity))

        command = (command_name, None, args, kwargs)

        response = worker.send_command(
            location, *command, return_ids=response_ids, return_value=return_value
        )

        return response

    return remote_exec
