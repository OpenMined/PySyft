# stdlib
from collections import defaultdict
import logging
import threading
import time
from typing import Any

# third party
from pydantic import BaseModel
from sherlock.lock import BaseLock

# relative
from ..serde.serializable import serializable

logger = logging.getLogger(__name__)
THREAD_FILE_LOCKS: dict[int, dict[str, int]] = defaultdict(dict)


@serializable(canonical_name="LockingConfig", version=1)
class LockingConfig(BaseModel):
    """
    Locking configuration.

    Attributes:
        lock_name (str): Lock name.
        namespace (str | None): Namespace to use for setting lock keys in the backend store.
        expire (int | None): Lock expiration time in seconds. If explicitly set to `None`, the lock will not expire.
        timeout (int | None): Timeout to acquire lock (in seconds).
        retry_interval (float): Retry interval to retry acquiring a lock if previous attempts failed.

    Args:
        lock_name (str): Lock name.
        namespace (str | None): Namespace to use for setting lock keys in the backend store.
        expire (int | None): Lock expiration time in seconds. If explicitly set to `None`, the lock will not expire.
        timeout (int | None): Timeout to acquire lock (in seconds).
        retry_interval (float): Retry interval to retry acquiring a lock if previous attempts failed.
    """

    lock_name: str = "syft_lock"
    namespace: str | None = None
    expire: int | None = 60
    timeout: int | None = 30
    retry_interval: float = 0.1


@serializable(canonical_name="NoLockingConfig", version=1)
class NoLockingConfig(LockingConfig):
    """
    No-locking policy.
    """

    pass


@serializable(canonical_name="ThreadingLockingConfig", version=1)
class ThreadingLockingConfig(LockingConfig):
    """
    Threading-based locking policy.
    """

    pass


class ThreadingLock(BaseLock):
    """
    Threading-based Lock. Used to provide the same API as the rest of the locks.
    """

    def __init__(self, expire: int, **kwargs: Any) -> None:
        self.expire = expire
        self.locked_timestamp: float = 0.0
        self.lock = threading.Lock()

    @property
    def _locked(self) -> bool:
        """
        Check if the lock has been acquired.

        Returns:
            bool: True if the lock is acquired, False otherwise.
        """
        locked = self.lock.locked()
        if (
            locked
            and time.time() - self.locked_timestamp >= self.expire
            and self.expire != -1
        ):
            self._release()

        return self.lock.locked()

    def _acquire(self) -> bool:
        """
        Acquire the lock in a non-blocking fashion.

        Returns:
            bool: True if the lock was successfully acquired, False otherwise.
        """
        locked = self.lock.locked()
        if (
            locked
            and time.time() - self.locked_timestamp > self.expire
            and self.expire != -1
        ):
            self._release()

        status = self.lock.acquire(blocking=False)
        if status:
            self.locked_timestamp = time.time()
        return status

    def _release(self) -> None:
        """
        Release the acquired lock.
        """
        try:
            return self.lock.release()
        except RuntimeError:  # already unlocked
            pass

    def _renew(self) -> bool:
        """
        Renew the acquired lock.

        Returns:
            bool: True if the lock was successfully renewed, False otherwise.
        """
        return True


class SyftLock(BaseLock):
    """
    Syft Lock implementation.

    Args:
        config (LockingConfig): Configuration specific to a locking strategy.

    Raises:
        ValueError: If an unsupported config type is provided.
    """

    def __init__(self, config: LockingConfig):
        self.config = config

        self.lock_name = config.lock_name
        self.namespace = config.namespace
        self.expire = config.expire
        self.timeout = config.timeout
        self.retry_interval = config.retry_interval

        self.passthrough = False

        self._lock: BaseLock | None = None

        base_params = {
            "lock_name": config.lock_name,
            "namespace": config.namespace,
            "expire": config.expire,
            "timeout": config.timeout,
            "retry_interval": config.retry_interval,
        }
        if isinstance(config, NoLockingConfig):
            self.passthrough = True
        elif isinstance(config, ThreadingLockingConfig):
            self._lock = ThreadingLock(**base_params)
        else:
            raise ValueError("Unsupported config type")

    @property
    def _locked(self) -> bool:
        """
        Check if the lock has been acquired.

        Returns:
            bool: True if the lock is acquired, False otherwise.
        """
        if self.passthrough:
            return False
        return self._lock._locked if self._lock else False

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire a lock, blocking or non-blocking.

        Args:
            blocking (bool): Acquire a lock in a blocking or non-blocking fashion. Defaults to True.

        Returns:
            bool: True if the lock was successfully acquired, False otherwise.
        """
        if not blocking:
            return self._acquire()

        timeout: float = float(self.timeout)
        start_time = time.time()
        elapsed: float = 0.0
        while timeout >= elapsed:
            if not self._acquire():
                time.sleep(self.retry_interval)
                elapsed = time.time() - start_time
            else:
                return True
        logger.debug(
            f"Timeout elapsed after {self.timeout} seconds while trying to acquiring lock."
        )
        return False

    def _acquire(self) -> bool:
        """
        Acquire the lock in a non-blocking fashion.

        Returns:
            bool: True if the lock was successfully acquired, False otherwise.
        """
        if self.passthrough:
            return True

        try:
            return self._lock._acquire() if self._lock else False
        except BaseException:
            return False

    def _release(self) -> bool | None:
        """
        Release the acquired lock.

        Returns:
            bool | None: True if the lock was successfully released, None otherwise.
        """
        if self.passthrough:
            return None
        if not self._lock:
            return None
        try:
            return self._lock._release()
        except BaseException:
            return None

    def _renew(self) -> bool:
        """
        Renew the acquired lock.

        Returns:
            bool: True if the lock was successfully renewed, False otherwise.
        """
        if self.passthrough:
            return True

        return self._lock._renew() if self._lock else False
