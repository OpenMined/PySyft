# stdlib
from threading import Thread
from typing import Tuple

# third party
from joblib import Parallel
from joblib import delayed
import pytest

# syft absolute
from syft.store.document_store import QueryKeys
from syft.store.sqlite_document_store import SQLiteStorePartition

# relative
from .store_fixtures_test import sqlite_store_partition_fn
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockSyftObject

REPEATS = 20


def test_sqlite_store_partition_sanity(
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    assert hasattr(sqlite_store_partition, "data")
    assert hasattr(sqlite_store_partition, "unique_keys")
    assert hasattr(sqlite_store_partition, "searchable_keys")


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_set(
    root_verify_key,
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    obj = MockSyftObject(data=1)
    res = sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert (
        len(
            sqlite_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    res = sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert res.is_err()
    assert (
        len(
            sqlite_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    res = sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=True)
    assert res.is_ok()
    assert (
        len(
            sqlite_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    obj2 = MockSyftObject(data=2)
    res = sqlite_store_partition.set(root_verify_key, obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert (
        len(
            sqlite_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 2
    )

    for idx in range(REPEATS):
        obj = MockSyftObject(data=idx)
        res = sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
        assert res.is_ok()
        assert (
            len(
                sqlite_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == 3 + idx
        )


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_delete(
    root_verify_key,
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    objs = []
    for v in range(REPEATS):
        obj = MockSyftObject(data=v)
        sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(
        sqlite_store_partition.all(
            root_verify_key,
        ).ok()
    ) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    res = sqlite_store_partition.delete(root_verify_key, key)
    assert res.is_err()
    assert len(
        sqlite_store_partition.all(
            root_verify_key,
        ).ok()
    ) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = sqlite_store_partition.settings.store_key.with_obj(v)
        res = sqlite_store_partition.delete(root_verify_key, key)
        assert res.is_ok()
        assert (
            len(
                sqlite_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == len(objs) - idx - 1
        )

        res = sqlite_store_partition.delete(root_verify_key, key)
        assert res.is_err()
        assert (
            len(
                sqlite_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == len(objs) - idx - 1
        )

    assert (
        len(
            sqlite_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 0
    )


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_update(
    root_verify_key,
    sqlite_store_partition: SQLiteStorePartition,
) -> None:
    # add item
    obj = MockSyftObject(data=1)
    sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert (
        len(
            sqlite_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = sqlite_store_partition.settings.store_key.with_obj(rand_obj)
    res = sqlite_store_partition.update(root_verify_key, key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(REPEATS):
        key = sqlite_store_partition.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = sqlite_store_partition.update(root_verify_key, key, obj_new)
        assert res.is_ok()

        # The ID should stay the same on update, unly the values are updated.
        assert (
            len(
                sqlite_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == 1
        )
        assert (
            sqlite_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .id
            == obj.id
        )
        assert (
            sqlite_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .id
            != obj_new.id
        )
        assert (
            sqlite_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .data
            == v
        )

        stored = sqlite_store_partition.get_all_from_store(
            root_verify_key, QueryKeys(qks=[key])
        )
        assert stored.ok()[0].data == v


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_set_threading(
    sqlite_workspace: Tuple,
    root_verify_key,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        sqlite_store_partition = sqlite_store_partition_fn(
            root_verify_key, sqlite_workspace
        )
        for idx in range(repeats):
            obj = MockObjectType(data=idx)

            for _ in range(10):
                res = sqlite_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok(), res

        return execution_err

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None

    sqlite_store_partition = sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace
    )
    stored_cnt = len(
        sqlite_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == thread_cnt * repeats


@pytest.mark.skip(reason="The tests are highly flaky, delaying progress on PR's")
def test_sqlite_store_partition_set_joblib(
    root_verify_key,
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    def _kv_cbk(tid: int) -> None:
        for idx in range(repeats):
            sqlite_store_partition = sqlite_store_partition_fn(
                root_verify_key, sqlite_workspace
            )
            obj = MockObjectType(data=idx)

            for _ in range(10):
                res = sqlite_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                return res

        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None

    sqlite_store_partition = sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace
    )
    stored_cnt = len(
        sqlite_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == thread_cnt * repeats


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_update_threading(
    root_verify_key,
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    sqlite_store_partition = sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace
    )
    obj = MockSyftObject(data=0)
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        sqlite_store_partition_local = sqlite_store_partition_fn(
            root_verify_key, sqlite_workspace
        )
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)

            for _ in range(10):
                res = sqlite_store_partition_local.update(root_verify_key, key, obj)
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok(), res

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_update_joblib(
    root_verify_key,
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    sqlite_store_partition = sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace
    )
    obj = MockSyftObject(data=0)
    key = sqlite_store_partition.settings.store_key.with_obj(obj)
    sqlite_store_partition.set(root_verify_key, obj, ignore_duplicates=False)

    def _kv_cbk(tid: int) -> None:
        sqlite_store_partition_local = sqlite_store_partition_fn(
            root_verify_key, sqlite_workspace
        )
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)

            for _ in range(10):
                res = sqlite_store_partition_local.update(root_verify_key, key, obj)
                if res.is_ok():
                    break

            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )

    for execution_err in errs:
        assert execution_err is None


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_set_delete_threading(
    root_verify_key,
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        sqlite_store_partition = sqlite_store_partition_fn(
            root_verify_key, sqlite_workspace
        )

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = sqlite_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok()

            key = sqlite_store_partition.settings.store_key.with_obj(obj)

            res = sqlite_store_partition.delete(root_verify_key, key)
            if res.is_err():
                execution_err = res
            assert res.is_ok(), res

    tids = []
    for tid in range(thread_cnt):
        thread = Thread(target=_kv_cbk, args=(tid,))
        thread.start()

        tids.append(thread)

    for thread in tids:
        thread.join()

    assert execution_err is None

    sqlite_store_partition = sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace
    )
    stored_cnt = len(
        sqlite_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == 0


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_sqlite_store_partition_set_delete_joblib(
    root_verify_key,
    sqlite_workspace: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    def _kv_cbk(tid: int) -> None:
        sqlite_store_partition = sqlite_store_partition_fn(
            root_verify_key, sqlite_workspace
        )

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = sqlite_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                return res

            key = sqlite_store_partition.settings.store_key.with_obj(obj)

            res = sqlite_store_partition.delete(root_verify_key, key)
            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )
    for execution_err in errs:
        assert execution_err is None

    sqlite_store_partition = sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace
    )
    stored_cnt = len(
        sqlite_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == 0
