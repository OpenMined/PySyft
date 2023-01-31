# type: ignore

# relative
from .backends import ShylockSyncBackend
from .exceptions import ShylockException


class Lock:
    """
    The best way to use Shylock. Stores references to backend and name for convenient use.

    >>> lock = Lock("my-lock")
    >>> try:
    >>>     lock.acquire()
    >>> finally:
    >>>     lock.release()
    >>> with lock:
    >>>     print("Locked")
    >>> print("Released")
    """

    def __init__(self, name: str, backend: ShylockSyncBackend = None):
        self.name = name
        # relative
        from .manager import BACKEND

        self._backend = BACKEND if backend is None else backend
        if not issubclass(self._backend.__class__, ShylockSyncBackend):
            raise ShylockException(
                "shylock.Lock requires a ShylockSyncBackend, did you mean to use shylock.aio.Lock?"
            )

        self._locked = False
        if self._backend is None:
            raise ShylockException(
                "No Shylock backend set, configure one with shylock.configure, "
                "or pass instace of shylock.ShylockBackend as argument to Lock()"
            )

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def acquire(self, block: bool = True) -> bool:
        """
        Try to acquire the lock - optionally block until available
        :param block: Wait until lock is available
        :return: If lock was successfully acquired - always True if blocking
        """
        res = self._backend.acquire(self.name, block)

        if res:
            self._locked = True

        return res

    def locked(self) -> bool:
        """
        Does the lock believe it's currently locked - does not check actual backend
        :return: Locked state
        """
        return self._locked

    def release(self) -> None:
        """
        Release the lock
        """
        if not self._locked:
            raise ShylockException(
                f"Trying to unlock {self.name} without locking it first."
            )
        self._backend.release(self.name)
        self._locked = False
