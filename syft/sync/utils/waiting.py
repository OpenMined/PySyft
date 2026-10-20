"""Poll loops for calls that wait for another party, such as a peer approval."""

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Seconds between polls. Each poll makes Drive requests; a person approving or a
# job running takes longer than this anyway.
DEFAULT_POLL_INTERVAL = 15

# Seconds a wait_until_* helper waits before it raises TimeoutError.
DEFAULT_WAIT_TIMEOUT = 300


def poll_until(
    check: Callable[[], Optional[T]], deadline: float, poll_interval: float
) -> Optional[T]:
    """Sleep, then call ``check``, until it returns a value that is not None.

    ``deadline`` is a ``time.monotonic()`` value. The last sleep is cut to end
    at it. Returns None when the deadline passes first.
    """
    while (remaining := deadline - time.monotonic()) > 0:
        time.sleep(min(poll_interval, remaining))
        if (result := check()) is not None:
            return result
    return None


def wait_for(
    find: Callable[[], Optional[T]],
    what: str,
    describe: Callable[[], str],
    timeout: float,
    poll_interval: float,
) -> T:
    """Return the first result of ``find`` that is not None.

    ``find`` runs at once, then after each poll. An error from ``find`` stops
    the wait. ``what`` names the thing waited for; ``describe`` tells the state
    now, for the messages.

    Raises:
        TimeoutError: ``find`` still returns None after ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    if (found := find()) is not None:
        return found
    if deadline > time.monotonic():
        print(f"Waiting up to {timeout:g}s for {what}: {describe()}")
        if (found := poll_until(find, deadline, poll_interval)) is not None:
            return found
    raise TimeoutError(
        f"Timed out after {timeout:g}s waiting for {what}: {describe()}."
    )
