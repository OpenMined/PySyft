# relative
from ..backends import ShylockAsyncBackend
from ..exceptions import ShylockException
from ..manager import BACKEND


class Lock:
    """
    The best way to use Shylock. Stores references to backend and name for convenient use.

    >>> lock = Lock("my-lock")
    >>> try:
    >>>     await lock.acquire()
    >>> finally:
    >>>     await lock.release()
    >>> async with lock:
    >>>     print("Locked")
    >>> print("Released")
    """

    def __init__(self, name: str, backend: ShylockAsyncBackend = None):
        self.name = name
        self._backend = BACKEND if backend is None else backend

        if not issubclass(self._backend.__class__, ShylockAsyncBackend):
            raise ShylockException(
                "shylock.aio.Lock requires a ShylockAsyncBackend, did you mean to use shylock.Lock?"
            )

        self._locked = False
        if self._backend is None:
            raise ShylockException(
                "No Shylock backend set, configure one with shylock.configure, "
                "or pass instace of shylock.ShylockBackend as argument to Lock()"
            )

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

    async def acquire(self, block: bool = True) -> bool:
        """
        Try to acquire the lock - optionally block until available
        :param block: Wait until lock is available
        :return: If lock was successfully acquired - always True if blocking
        """
        res = await self._backend.acquire(self.name, block)

        if res:
            self._locked = True

        return res

    async def locked(self) -> bool:
        """
        Does the lock believe it's currently locked - does not check actual backend
        :return: Locked state
        """
        return self._locked

    async def release(self):
        """
        Release the lock
        """
        if not self._locked:
            raise ShylockException(
                f"Trying to unlock {self.name} without locking it first."
            )
        await self._backend.release(self.name)
        self._locked = False
