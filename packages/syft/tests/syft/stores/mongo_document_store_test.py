# stdlib
import sys
from threading import Thread
from typing import Tuple

# third party
from joblib import Parallel
from joblib import delayed
import pytest

# syft absolute
from syft.store.document_store import PartitionSettings
from syft.store.document_store import QueryKeys
from syft.store.mongo_client import MongoStoreClientConfig
from syft.store.mongo_document_store import MongoStoreConfig
from syft.store.mongo_document_store import MongoStorePartition

# relative
from .store_constants_test import generate_db_name
from .store_fixtures_test import mongo_store_partition_fn
from .store_mocks_test import MockObjectType
from .store_mocks_test import MockSyftObject

REPEATS = 20


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_sanity(
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    assert hasattr(mongo_store_partition, "_collection")


def test_mongo_store_partition_init_failed(root_verify_key) -> None:
    # won't connect
    mongo_config = MongoStoreClientConfig(connectTimeoutMS=1, timeoutMS=1)

    store_config = MongoStoreConfig(client_config=mongo_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = MongoStorePartition(
        root_verify_key, settings=settings, store_config=store_config
    )

    res = store.init_store()
    assert res.is_err()


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_set(
    root_verify_key, mongo_store_partition: MongoStorePartition
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    obj = MockSyftObject(data=1)

    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)

    assert res.is_ok()
    assert res.ok() == obj
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert res.is_err()
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=True)
    assert res.is_ok()
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    obj2 = MockSyftObject(data=2)
    res = mongo_store_partition.set(root_verify_key, obj2, ignore_duplicates=False)
    assert res.is_ok()
    assert res.ok() == obj2
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 2
    )

    for idx in range(REPEATS):
        obj = MockSyftObject(data=idx)
        res = mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
        assert res.is_ok()
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == 3 + idx
        )


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_delete(
    root_verify_key,
    mongo_store_partition: MongoStorePartition,
) -> None:
    res = mongo_store_partition.init_store()
    assert res.is_ok()

    objs = []
    for v in range(REPEATS):
        obj = MockSyftObject(data=v)
        mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
        objs.append(obj)

    assert len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    ) == len(objs)

    # random object
    obj = MockSyftObject(data="bogus")
    key = mongo_store_partition.settings.store_key.with_obj(obj)
    res = mongo_store_partition.delete(root_verify_key, key)
    assert res.is_err()
    assert len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    ) == len(objs)

    # cleanup store
    for idx, v in enumerate(objs):
        key = mongo_store_partition.settings.store_key.with_obj(v)
        res = mongo_store_partition.delete(root_verify_key, key)
        assert res.is_ok()
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == len(objs) - idx - 1
        )

        res = mongo_store_partition.delete(root_verify_key, key)
        assert res.is_err()
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == len(objs) - idx - 1
        )

    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 0
    )


@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
def test_mongo_store_partition_update(
    root_verify_key,
    mongo_store_partition: MongoStorePartition,
) -> None:
    mongo_store_partition.init_store()

    # add item
    obj = MockSyftObject(data=1)
    mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    assert (
        len(
            mongo_store_partition.all(
                root_verify_key,
            ).ok()
        )
        == 1
    )

    # fail to update missing keys
    rand_obj = MockSyftObject(data="bogus")
    key = mongo_store_partition.settings.store_key.with_obj(rand_obj)
    res = mongo_store_partition.update(root_verify_key, key, obj)
    assert res.is_err()

    # update the key multiple times
    for v in range(REPEATS):
        key = mongo_store_partition.settings.store_key.with_obj(obj)
        obj_new = MockSyftObject(data=v)

        res = mongo_store_partition.update(root_verify_key, key, obj_new)
        assert res.is_ok()

        # The ID should stay the same on update, unly the values are updated.
        assert (
            len(
                mongo_store_partition.all(
                    root_verify_key,
                ).ok()
            )
            == 1
        )
        assert (
            mongo_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .id
            == obj.id
        )
        assert (
            mongo_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .id
            != obj_new.id
        )
        assert (
            mongo_store_partition.all(
                root_verify_key,
            )
            .ok()[0]
            .data
            == v
        )

        stored = mongo_store_partition.get_all_from_store(
            root_verify_key, QueryKeys(qks=[key])
        )
        assert stored.ok()[0].data == v


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_set_threading(
    root_verify_key,
    mongo_server_mock: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    execution_err = None
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        mongo_store_partition = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )
        for idx in range(repeats):
            obj = MockObjectType(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
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

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == thread_cnt * repeats


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_set_joblib(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        for idx in range(repeats):
            mongo_store_partition = mongo_store_partition_fn(
                root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
            )
            obj = MockObjectType(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
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

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == thread_cnt * repeats


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.xfail(reason="Fails in CI sometimes")
def test_mongo_store_partition_update_threading(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()
    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )

    obj = MockSyftObject(data=0)
    key = mongo_store_partition.settings.store_key.with_obj(obj)
    mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)
    execution_err = None

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err

        mongo_store_partition_local = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)

            for _ in range(10):
                res = mongo_store_partition_local.update(root_verify_key, key, obj)
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


@pytest.mark.xfail(reason="SyftObjectRegistry does only in-memory caching")
@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_update_joblib(
    root_verify_key,
    mongo_server_mock: Tuple,
) -> None:
    thread_cnt = 3
    repeats = REPEATS

    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    obj = MockSyftObject(data=0)
    key = mongo_store_partition.settings.store_key.with_obj(obj)
    mongo_store_partition.set(root_verify_key, obj, ignore_duplicates=False)

    def _kv_cbk(tid: int) -> None:
        mongo_store_partition_local = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )
        for repeat in range(repeats):
            obj = MockSyftObject(data=repeat)

            for _ in range(10):
                res = mongo_store_partition_local.update(root_verify_key, key, obj)
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


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_set_delete_threading(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    execution_err = None
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        nonlocal execution_err
        mongo_store_partition = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                execution_err = res
            assert res.is_ok()

            key = mongo_store_partition.settings.store_key.with_obj(obj)

            res = mongo_store_partition.delete(root_verify_key, key)
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

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == 0


@pytest.mark.skipif(
    sys.platform != "linux", reason="pytest_mock_resources + docker issues on Windows"
)
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_mongo_store_partition_set_delete_joblib(
    root_verify_key,
    mongo_server_mock,
) -> None:
    thread_cnt = 3
    repeats = REPEATS
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    def _kv_cbk(tid: int) -> None:
        mongo_store_partition = mongo_store_partition_fn(
            root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
        )

        for idx in range(repeats):
            obj = MockSyftObject(data=idx)

            for _ in range(10):
                res = mongo_store_partition.set(
                    root_verify_key, obj, ignore_duplicates=False
                )
                if res.is_ok():
                    break

            if res.is_err():
                return res

            key = mongo_store_partition.settings.store_key.with_obj(obj)

            res = mongo_store_partition.delete(root_verify_key, key)
            if res.is_err():
                return res
        return None

    errs = Parallel(n_jobs=thread_cnt)(
        delayed(_kv_cbk)(idx) for idx in range(thread_cnt)
    )
    for execution_err in errs:
        assert execution_err is None

    mongo_store_partition = mongo_store_partition_fn(
        root_verify_key, mongo_db_name=mongo_db_name, **mongo_kwargs
    )
    stored_cnt = len(
        mongo_store_partition.all(
            root_verify_key,
        ).ok()
    )
    assert stored_cnt == 0
