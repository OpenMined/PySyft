import time
from functools import wraps


def assert_time(max_time):
    """
    Decorator used to assert time execution of functions.
    Args:
        max_time: int or float. Maximum time in seconds that the decorated
        function should take to be executed
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            func(*args, **kwargs)
            dt = time.time() - t0
            assert dt < max_time, f"Test run in {round(dt, 2)} > {round(max_time, 2)} s"

        return wrapper

    return decorate
