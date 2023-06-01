# stdlib
import datetime
from pathlib import Path
import random
import string
import sys
import tempfile
from threading import Thread
import time

# third party
from joblib import Parallel
from joblib import delayed
import pytest
from pytest_mock_resources import create_redis_fixture

# syft absolute
from syft.store.locks import FileLockingConfig
from syft.store.locks import LockingConfig
from syft.store.locks import NoLockingConfig
from syft.store.locks import RedisLockingConfig
from syft.store.locks import SyftLock
from syft.store.locks import ThreadingLockingConfig

redis_server_mock = create_redis_fixture(scope="session")

def_params = {
    "lock_name": "testing_lock",
    "expire": 5,  # seconds,
    "timeout": 1,  # seconds,
    "retry_interval": 0.1,  # seconds,
}


def generate_lock_name(length: int = 10) -> str:
    random.seed(datetime.datetime.now().timestamp())
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


@pytest.fixture(scope="function")
def locks_nop_config(request):
    def_params["lock_name"] = generate_lock_name()
    return NoLockingConfig(**def_params)


@pytest.fixture(scope="function")
def locks_threading_config(request):
    def_params["lock_name"] = generate_lock_name()
    return ThreadingLockingConfig(**def_params)


@pytest.fixture(scope="function")
def locks_file_config():
    def_params["lock_name"] = generate_lock_name()
    return FileLockingConfig(**def_params)


@pytest.fixture(scope="function")
def locks_redis_config(redis_server_mock):
    def_params["lock_name"] = generate_lock_name()
    redis_config = redis_server_mock.pmr_credentials.as_redis_kwargs()
    return RedisLockingConfig(**def_params, client=redis_config)


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_nop_config"),
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
def test_sanity(config: LockingConfig):
    lock = SyftLock(config)

    assert lock is not None


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_nop_config"),
    ],
)
def test_acquire_nop(config: LockingConfig):
    lock = SyftLock(config)

    assert lock.locked() is False

    acq_ok = lock.acquire()
    assert acq_ok

    assert lock.locked() is False

    lock.release()

    assert lock.locked() is False


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_release(config: LockingConfig):
    lock = SyftLock(config)

    expected_not_locked = lock.locked()

    acq_ok = lock.acquire()
    assert acq_ok

    expected_locked = lock.locked()

    lock.release()

    expected_not_locked_again = lock.locked()

    assert not expected_not_locked
    assert expected_locked
    assert not expected_not_locked_again


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_release_with(config: LockingConfig):
    was_locked = True
    with SyftLock(config) as lock:
        was_locked = lock.locked()

    assert was_locked


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_expire(config: LockingConfig):
    config.expire = 1  # second
    lock = SyftLock(config)

    expected_not_locked = lock.locked()

    acq_ok = lock.acquire(blocking=True)
    assert acq_ok

    expected_locked = lock.locked()

    time.sleep(config.expire + 0.1)

    expected_not_locked_again = lock.locked()

    assert not expected_not_locked
    assert expected_locked
    assert not expected_not_locked_again


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_double_aqcuire_timeout_fail(config: LockingConfig):
    config.timeout = 1
    config.expire = 5
    lock = SyftLock(config)

    acq_ok = lock.acquire(blocking=True)
    assert acq_ok

    not_acq = lock.acquire(blocking=True)

    lock.release()

    assert not not_acq


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_double_aqcuire_timeout_ok(config: LockingConfig):
    config.timeout = 2
    config.expire = 1
    lock = SyftLock(config)

    lock.locked()

    acq_ok = lock.acquire(blocking=True)
    assert acq_ok

    also_acq = lock.acquire(blocking=True)

    lock.release()

    assert also_acq


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_double_aqcuire_nonblocking(config: LockingConfig):
    config.timeout = 2
    config.expire = 1
    lock = SyftLock(config)

    lock.locked()

    acq_ok = lock.acquire(blocking=False)
    assert acq_ok

    not_acq = lock.acquire(blocking=False)

    lock.release()

    assert not not_acq


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_double_aqcuire_retry_interval(config: LockingConfig):
    config.timeout = 2
    config.expire = 1
    config.retry_interval = 3
    lock = SyftLock(config)

    lock.locked()

    acq_ok = lock.acquire(blocking=True)
    assert acq_ok

    not_acq = lock.acquire(blocking=True)

    lock.release()

    assert not not_acq


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_double_release(config: LockingConfig):
    lock = SyftLock(config)

    lock.acquire(blocking=True)

    lock.release()
    lock.release()


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_acquire_same_name_diff_namespace(config: LockingConfig):
    config.namespace = "ns1"
    lock1 = SyftLock(config)
    assert lock1.acquire(blocking=True)

    config.namespace = "ns2"
    lock2 = SyftLock(config)
    assert lock2.acquire(blocking=True)

    lock2.release()
    lock1.release()


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_locks_parallel_multithreading(config: LockingConfig) -> None:
    thread_cnt = 3
    repeats = 100

    temp_dir = Path(tempfile.TemporaryDirectory().name)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "dbg.txt"
    if temp_file.exists():
        temp_file.unlink()

    with open(temp_file, "w") as f:
        f.write("0")

    config.timeout = 10
    lock = SyftLock(config)

    def _kv_cbk(tid: int) -> None:
        for _idx in range(repeats):
            locked = lock.acquire()
            if not locked:
                continue

            for _retry in range(10):
                try:
                    with open(temp_file, "r") as f:
                        prev = f.read()
                        prev = int(prev)
                    with open(temp_file, "w") as f:
                        f.write(str(prev + 1))
                        f.flush()
                    break
                except BaseException as e:
                    print("failed ", e)

            lock.release()

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    with open(temp_file) as f:
        stored = int(f.read())

    assert stored == thread_cnt * repeats


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_file_config"),
        pytest.lazy_fixture("locks_redis_config"),
    ],
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_parallel_joblib(
    config: LockingConfig,
) -> None:
    thread_cnt = 3
    repeats = 100

    temp_dir = Path(tempfile.TemporaryDirectory().name)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "dbg.txt"
    if temp_file.exists():
        temp_file.unlink()

    with open(temp_file, "w") as f:
        f.write("0")

    def _kv_cbk(tid: int) -> None:
        for _idx in range(repeats):
            with SyftLock(config):
                with open(temp_file, "r") as f:
                    prev = int(f.read())
                with open(temp_file, "w") as f:
                    f.write(str(prev + 1))

    Parallel(n_jobs=thread_cnt)(delayed(_kv_cbk)(idx) for idx in range(thread_cnt))

    with open(temp_file) as f:
        stored = int(f.read())

    assert stored == thread_cnt * repeats
