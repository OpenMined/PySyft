# stdlib
import tempfile
import time
from pathlib import Path
from secrets import token_hex
from threading import Thread

# third party
import pytest

# syft absolute
from syft.store.locks import (
    LockingConfig,
    NoLockingConfig,
    SyftLock,
    ThreadingLockingConfig,
)

def_params = {
    "lock_name": "testing_lock",
    "expire": 5,  # seconds,
    "timeout": 1,  # seconds,
    "retry_interval": 0.1,  # seconds,
}


@pytest.fixture()
def locks_nop_config(request):
    def_params["lock_name"] = token_hex(8)
    return NoLockingConfig(**def_params)


@pytest.fixture()
def locks_threading_config(request):
    def_params["lock_name"] = token_hex(8)
    return ThreadingLockingConfig(**def_params)


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_nop_config"),
        pytest.lazy_fixture("locks_threading_config"),
    ],
)
def test_sanity(config: LockingConfig) -> None:
    lock = SyftLock(config)

    assert lock is not None


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_nop_config"),
    ],
)
def test_acquire_nop(config: LockingConfig) -> None:
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
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_release(config: LockingConfig) -> None:
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
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_release_with(config: LockingConfig) -> None:
    was_locked = True
    with SyftLock(config) as lock:
        was_locked = lock.locked()

    assert was_locked


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
    ],
)
def test_acquire_expire(config: LockingConfig) -> None:
    config.expire = 1  # second
    lock = SyftLock(config)

    expected_not_locked = lock.locked()

    acq_ok = lock.acquire(blocking=True)
    assert acq_ok

    expected_locked = lock.locked()

    time.sleep(config.expire + 1.0)

    expected_not_locked_again = lock.locked()

    assert not expected_not_locked
    assert expected_locked
    assert not expected_not_locked_again


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_double_aqcuire_timeout_fail(config: LockingConfig) -> None:
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
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_double_aqcuire_timeout_ok(config: LockingConfig) -> None:
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
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_double_aqcuire_nonblocking(config: LockingConfig) -> None:
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
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_double_aqcuire_retry_interval(config: LockingConfig) -> None:
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
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_double_release(config: LockingConfig) -> None:
    lock = SyftLock(config)

    lock.acquire(blocking=True)

    lock.release()
    lock.release()


@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
    ],
)
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_acquire_same_name_diff_namespace(config: LockingConfig) -> None:
    config.namespace = "ns1"
    lock1 = SyftLock(config)
    assert lock1.acquire(blocking=True)

    config.namespace = "ns2"
    lock2 = SyftLock(config)
    assert lock2.acquire(blocking=True)

    lock2.release()
    lock1.release()


@pytest.mark.skip(reason="The tests are highly flaky, delaying progress on PR's")
@pytest.mark.parametrize(
    "config",
    [
        pytest.lazy_fixture("locks_threading_config"),
    ],
)
def test_locks_parallel_multithreading(config: LockingConfig) -> None:
    thread_cnt = 3
    repeats = 5

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
                    with open(temp_file) as f:
                        prev = f.read()
                        prev = int(prev)
                    with open(temp_file, "w") as f:
                        f.write(str(prev + 1))
                        f.flush()
                    break
                except BaseException:
                    pass

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
