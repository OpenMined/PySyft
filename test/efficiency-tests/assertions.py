import time
from functools import wraps


def assert_time(time):
    """
    Decorator used to assert time execution of functions.
    Args:
        time: int or float. Maximum time in seconds that the decorated
        function should take to be executed
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            func(*args, **kwargs)
            dt = time.time() - t0
            assert dt < time

        return wrapper

    return decorate
