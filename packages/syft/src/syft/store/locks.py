# stdlib
from collections import defaultdict
import datetime
import json
from pathlib import Path
import threading
import time
from typing import Callable
from typing import Dict
from typing import Optional
import uuid

# third party
from pydantic import BaseModel
import redis
from sherlock.lock import BaseLock
from sherlock.lock import FileLock
from sherlock.lock import RedisLock

# relative
from ..serde.serializable import serializable

THREAD_FILE_LOCKS: Dict[int, Dict[str, int]] = defaultdict(dict)


@serializable()
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


@serializable()
class NoLockingConfig(LockingConfig):
    """
    No-locking policy
    """

    pass


@serializable()
class ThreadingLockingConfig(LockingConfig):
    """
    Threading-based locking policy
    """

    pass


@serializable()
class FileLockingConfig(LockingConfig):
    """File locking policy"""

    client_path: Optional[Path] = None


@serializable()
class RedisClientConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = None


@serializable()
class RedisLockingConfig(LockingConfig):
    """Redis locking policy"""

    client: RedisClientConfig = RedisClientConfig()


class ThreadingLock(BaseLock):
    """
    Threading-based Lock. Used to provide the same API as the rest of the locks.
    """

    def __init__(self, expire: int, **kwargs):
        self.expire = expire
        self.locked_timestamp = 0
        self.lock = threading.Lock()

    @property
    def _locked(self):
        """
        Implementation of method to check if lock has been acquired. Must be
        :returns: if the lock is acquired or not
        :rtype: bool
        """
        locked = self.lock.locked()
        if (
            locked
            and time.time() - self.locked_timestamp >= self.expire
            and self.expire != -1
        ):
            self._release()

        return self.lock.locked()

    def _acquire(self):
        """
        Implementation of acquiring a lock in a non-blocking fashion.
        :returns: if the lock was successfully acquired or not
        :rtype: bool
        """
        locked = self.lock.locked()
        if (
            locked
            and time.time() - self.locked_timestamp > self.expire
            and self.expire != -1
        ):
            self._release()

        status = self.lock.acquire(
            blocking=False
        )  # timeout/retries handle in the `acquire` method
        if status:
            self.locked_timestamp = time.time()
        return status

    def _release(self):
        """
        Implementation of releasing an acquired lock.
        """

        try:
            return self.lock.release()
        except RuntimeError:  # already unlocked
            pass

    def _renew(self) -> bool:
        """
        Implementation of renewing an acquired lock.
        """
        return True


class PatchedFileLock(FileLock):
    """
    Implementation of lock with the file system as the backend for synchronization.
    This version patches for the `FileLock._expiry_time` crash(https://github.com/py-sherlock/sherlock/issues/71)

    `sherlock.FileLock` might not work as expected for Python threads.
    It uses re-entrant OS locks, meaning that multiple Python threads could acquire the lock at the same time.
    For different processes/OS threads, the file lock will work as expected.
    We need to patch the lock to handle Python threads too.

    """

    def __init__(self, *args, **kwargs) -> None:
        self._lock_file_enabled = True
        try:
            super().__init__(*args, **kwargs)
        except BaseException as e:
            print(f"Failed to create a file lock = {e}. Using memory-lock only")
            self._lock_file_enabled = False

        self._lock_py_thread = ThreadingLock(*args, **kwargs)

    def _expiry_time(self) -> str:
        if self.expire is not None:
            expiry_time = self._now() + datetime.timedelta(seconds=self.expire)
        else:
            expiry_time = datetime.datetime.max.replace(
                tzinfo=datetime.timezone.utc
            ).astimezone(datetime.timezone.utc)
        return expiry_time.isoformat()

    def _thread_safe_cbk(self, cbk: Callable) -> bool:
        # Acquire lock at Python level(if-needed)
        locked = self._lock_py_thread._acquire()
        if not locked:
            return False

        try:
            result = cbk()
        except BaseException as e:
            print(e)
            result = False

        self._lock_py_thread._release()
        return result

    def _acquire(self) -> bool:
        return self._thread_safe_cbk(self._acquire_file_lock)

    def _release(self) -> None:
        res = self._thread_safe_cbk(self._release_file_lock)
        return res

    def _acquire_file_lock(self) -> bool:
        if not self._lock_file_enabled:
            return True

        owner = str(uuid.uuid4())

        # Acquire lock at OS level
        with self._lock_file:
            if self._data_file.exists():
                for _retry in range(10):
                    try:
                        data = json.loads(self._data_file.read_text())
                        break
                    except BaseException:
                        time.sleep(0.1)
                    if _retry == 9:
                        pass

                now = self._now()
                has_expired = self._has_expired(data, now)
                if owner != data["owner"]:
                    if not has_expired:
                        # Someone else holds the lock.
                        return False
                    else:
                        # Lock is available for us to take.
                        data = {"owner": owner, "expiry_time": self._expiry_time()}
                else:
                    # Same owner so do not set or modify Lease.
                    return False
            else:
                data = {"owner": owner, "expiry_time": self._expiry_time()}

            # Write new data back to file.
            self._data_file.touch()
            self._data_file.write_text(json.dumps(data))

            # We succeeded in writing to the file so we now hold the lock.
            self._owner = owner

            return True

    @property
    def _locked(self):
        if self._lock_py_thread.locked():
            return True

        if not self._lock_file_enabled:
            return False

        if not self._data_file.exists():
            # File doesn't exist so can't be locked.
            return False

        with self._lock_file:
            data = None
            for _retry in range(10):
                try:
                    data = json.loads(self._data_file.read_text())
                    break
                except BaseException:
                    time.sleep(0.1)

        if data is None:
            return False

        if self._has_expired(data, self._now()):
            # File exists but has expired.
            return False

        # Lease exists and has not expired.
        return True

    def _release_file_lock(self) -> None:
        if not self._lock_file_enabled:
            return

        if self._owner is None:
            return

        if not self._data_file.exists():
            return

        with self._lock_file:
            data = None
            for _retry in range(10):
                try:
                    data = json.loads(self._data_file.read_text())
                    break
                except BaseException:
                    time.sleep(0.1)

            if data is None:
                return

            if self._owner == data["owner"]:
                self._data_file.unlink()
                self._owner = None


class SyftLock(BaseLock):
    """
    Syft Lock implementations.

    Params:
        config: Config specific to a locking strategy.
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
        elif isinstance(config, ThreadingLockingConfig):
            self._lock = ThreadingLock(**base_params)
        elif isinstance(config, FileLockingConfig):
            client = config.client_path
            self._lock = PatchedFileLock(
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
        Implementation of method to check if lock has been acquired.

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
        print(
            f"Timeout elapsed after {self.timeout} seconds while trying to acquiring lock."
        )
        # third party
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
