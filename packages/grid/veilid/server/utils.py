# stdlib
import asyncio
from collections.abc import Callable
from functools import wraps
import random
from typing import Any


def retry(
    exceptions: tuple[type[BaseException], ...] | type[BaseException],
    tries: int = 3,
    delay: int = 1,
    backoff: int = 2,
) -> Callable:
    """Retry calling the decorated function using exponential backoff.

    Args:
        exceptions (Tuple or Exception): The exception(s) to catch. Can be a tuple of exceptions or a single exception.
        tries (int): The maximum number of times to try the function (default: 3).
        delay (int): The initial delay between retries in seconds (default: 1).
        backoff (int): The exponential backoff factor (default: 2).

    Returns:
        The result of the decorated function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay: int = delay
            for _ in range(tries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    print(
                        f"Caught exception: {e}. Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            # Retry one last time before raising the exception
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def generate_random_alphabets(length: int) -> str:
    return "".join([random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length)])
