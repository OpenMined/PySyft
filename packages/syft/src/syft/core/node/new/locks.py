# stdlib
from pathlib import Path
import time
from typing import Optional

# third party
from pydantic import BaseModel
import redis
from sherlock.lock import BaseLock
from sherlock.lock import FileLock
from sherlock.lock import RedisLock

# relative
from ....logger import debug


class LockingConfig(BaseModel):
    """
    Locking config

    Args:
        lock_name: str
            Lock name
        namespace: Optional[str]
            Namespace to use for setting lock keys in the backend store.
        expire: Optional[int]
            Lock expiration time in seconds. If explicitly set to `None`, lock will not expire.
        timeout: Optional[int]
             Timeout to acquire lock(seconds)
        retry_interval: float
            Retry interval to retry acquiring a lock if previous attempts failed.
    """

    lock_name: str = "syft_lock"
    namespace: Optional[str] = None
    expire: Optional[int] = 60
    timeout: Optional[int] = 30
    retry_interval: float = 0.1


class NoLockingConfig(LockingConfig):
    """
    No-locking policy
    """

    pass


class FileLockingConfig(LockingConfig):
    """File locking policy"""

    client_path: Optional[Path] = None


class RedisClientConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = None


class RedisLockingConfig(LockingConfig):
    """Redis locking policy"""

    client: RedisClientConfig = RedisClientConfig()


class SyftLock(BaseLock):
    """
    Syft Lock implementations.
    """

    def __init__(self, config: LockingConfig):
        self.config = config

        self.lock_name = config.lock_name
        self.namespace = config.namespace
        self.expire = config.expire
        self.timeout = config.timeout
        self.retry_interval = config.retry_interval

        self.passthrough = False

        self._lock: Optional[BaseLock] = None

        base_params = {
            "lock_name": config.lock_name,
            "namespace": config.namespace,
            "expire": config.expire,
            "timeout": config.timeout,
            "retry_interval": config.retry_interval,
        }
        if isinstance(config, NoLockingConfig):
            self.passthrough = True
        elif isinstance(config, FileLockingConfig):
            client = config.client_path
            self._lock = FileLock(
                **base_params,
                client=client,
            )
        elif isinstance(config, RedisLockingConfig):
            client = redis.StrictRedis(**config.client.dict())

            self._lock = RedisLock(
                **base_params,
                client=client,
            )
        else:
            raise ValueError("Unsupported config type")

    @property
    def _locked(self):
        """
        Implementation of method to check if lock has been acquired. i

        :returns: if the lock is acquired or not
        :rtype: bool
        """
        if self.passthrough:
            return False

        return self._lock.locked()

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire a lock, blocking or non-blocking.
        :param bool blocking: acquire a lock in a blocking or non-blocking
                              fashion. Defaults to True.
        :returns: if the lock was successfully acquired or not
        :rtype: bool
        """

        if not blocking:
            return self._acquire()

        timeout = self.timeout
        start_time = time.time()
        elapsed = 0
        while timeout >= elapsed:
            if not self._acquire():
                time.sleep(self.retry_interval)
                elapsed = time.time() - start_time
            else:
                return True
        debug(
            "Timeout elapsed after %s seconds "
            "while trying to acquiring "
            "lock." % self.timeout
        )
        return False

    def _acquire(self) -> bool:
        """
        Implementation of acquiring a lock in a non-blocking fashion.
        `acquire` makes use of this implementation to provide blocking and non-blocking implementations.

        :returns: if the lock was successfully acquired or not
        :rtype: bool
        """
        if self.passthrough:
            return True

        try:
            return self._lock._acquire()
        except BaseException:
            return False

    def _release(self):
        """
        Implementation of releasing an acquired lock.
        """
        if self.passthrough:
            return

        try:
            return self._lock._release()
        except BaseException:
            pass

    def _renew(self) -> bool:
        """
        Implementation of renewing an acquired lock.
        """
        if self.passthrough:
            return True

        return self._lock._renew()
