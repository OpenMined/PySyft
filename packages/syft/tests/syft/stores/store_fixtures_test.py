# stdlib
from pathlib import Path
import tempfile
from typing import Generator
from typing import Tuple

# third party
from pymongo import MongoClient
import pytest
from pytest_mock_resources import create_mongo_fixture

# syft absolute
from syft.core.node.new.action_store import DictActionStore
from syft.core.node.new.action_store import SQLiteActionStore
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.dict_document_store import DictDocumentStore
from syft.core.node.new.dict_document_store import DictStoreConfig
from syft.core.node.new.dict_document_store import DictStorePartition
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.locks import FileLockingConfig
from syft.core.node.new.locks import LockingConfig
from syft.core.node.new.locks import NoLockingConfig
from syft.core.node.new.locks import ThreadingLockingConfig
from syft.core.node.new.mongo_client import MongoStoreClientConfig
from syft.core.node.new.mongo_document_store import MongoDocumentStore
from syft.core.node.new.mongo_document_store import MongoStoreConfig
from syft.core.node.new.mongo_document_store import MongoStorePartition
from syft.core.node.new.queue_stash import QueueStash
from syft.core.node.new.sqlite_document_store import SQLiteDocumentStore
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig
from syft.core.node.new.sqlite_document_store import SQLiteStorePartition

# relative
from .store_constants_test import generate_db_name
from .store_constants_test import sqlite_workspace_folder
from .store_constants_test import test_verify_key_string_root
from .store_mocks_test import MockObjectType

mongo_server_mock = create_mongo_fixture(scope="session")

locking_scenarios = ["nop", "file", "threading"]


def str_to_locking_config(conf: str) -> LockingConfig:
    if conf == "nop":
        return NoLockingConfig()
    elif conf == "file":
        lock_name = generate_db_name()

        temp_dir = tempfile.TemporaryDirectory().name

        workspace_folder = Path(temp_dir) / "filelock"
        workspace_folder.mkdir(parents=True, exist_ok=True)

        client_path = workspace_folder / lock_name

        return FileLockingConfig(client_path=client_path)
    elif conf == "threading":
        return ThreadingLockingConfig()
    else:
        raise NotImplementedError(f"unknown locking config {conf}")


@pytest.fixture(scope="function")
def sqlite_workspace() -> Generator:
    sqlite_db_name = generate_db_name()

    sqlite_workspace_folder.mkdir(parents=True, exist_ok=True)
    db_path = sqlite_workspace_folder / sqlite_db_name

    if db_path.exists():
        db_path.unlink()

    yield sqlite_workspace_folder, sqlite_db_name

    if db_path.exists():
        try:
            db_path.unlink()
        except BaseException as e:
            print("failed to cleanup sqlite db", e)


def sqlite_store_partition_fn(
    sqlite_workspace: Tuple[Path, str], locking_config_name: str = "nop"
):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)

    locking_config = str_to_locking_config(locking_config_name)
    store_config = SQLiteStoreConfig(
        client_config=sqlite_config, locking_config=locking_config
    )

    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(settings=settings, store_config=store_config)

    res = store.init_store()
    assert res.is_ok()

    return store


@pytest.fixture(scope="function", params=locking_scenarios)
def sqlite_store_partition(sqlite_workspace: Tuple[Path, str], request):
    locking_config_name = request.param
    return sqlite_store_partition_fn(
        sqlite_workspace, locking_config_name=locking_config_name
    )


def sqlite_document_store_fn(
    sqlite_workspace: Tuple[Path, str], locking_config_name: str = "nop"
):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)

    locking_config = str_to_locking_config(locking_config_name)
    store_config = SQLiteStoreConfig(
        client_config=sqlite_config, locking_config=locking_config
    )

    return SQLiteDocumentStore(store_config=store_config)


@pytest.fixture(scope="function", params=locking_scenarios)
def sqlite_document_store(sqlite_workspace: Tuple[Path, str], request):
    locking_config_name = request.param
    return sqlite_document_store_fn(
        sqlite_workspace, locking_config_name=locking_config_name
    )


def sqlite_queue_stash_fn(
    sqlite_workspace: Tuple[Path, str], locking_config_name: str = "nop"
):
    store = sqlite_document_store_fn(
        sqlite_workspace, locking_config_name=locking_config_name
    )
    return QueueStash(store=store)


@pytest.fixture(scope="function", params=locking_scenarios)
def sqlite_queue_stash(sqlite_workspace: Tuple[Path, str], request):
    locking_config_name = request.param
    return sqlite_queue_stash_fn(
        sqlite_workspace, locking_config_name=locking_config_name
    )


@pytest.fixture(scope="function", params=locking_scenarios)
def sqlite_action_store(sqlite_workspace: Tuple[Path, str], request):
    workspace, db_name = sqlite_workspace
    locking_config_name = request.param

    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)

    locking_config = str_to_locking_config(locking_config_name)
    store_config = SQLiteStoreConfig(
        client_config=sqlite_config, locking_config=locking_config
    )

    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return SQLiteActionStore(store_config=store_config, root_verify_key=ver_key)


def mongo_store_partition_fn(
    mongo_db_name: str = "mongo_db", locking_config_name: str = "nop", **mongo_kwargs
):
    mongo_client = MongoClient(**mongo_kwargs)
    mongo_config = MongoStoreClientConfig(client=mongo_client)

    locking_config = str_to_locking_config(locking_config_name)

    store_config = MongoStoreConfig(
        client_config=mongo_config, db_name=mongo_db_name, locking_config=locking_config
    )
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return MongoStorePartition(settings=settings, store_config=store_config)


@pytest.fixture(scope="function", params=locking_scenarios)
def mongo_store_partition(mongo_server_mock, request):
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()
    locking_config_name = request.param

    yield mongo_store_partition_fn(
        mongo_db_name=mongo_db_name,
        locking_config_name=locking_config_name,
        **mongo_kwargs,
    )

    # cleanup db
    try:
        mongo_client = MongoClient(**mongo_kwargs)
        mongo_client.drop_database(mongo_db_name)
    except BaseException as e:
        print("failed to cleanup mongo fixture", e)


def mongo_document_store_fn(
    mongo_db_name: str = "mongo_db", locking_config_name: str = "nop", **mongo_kwargs
):
    locking_config = str_to_locking_config(locking_config_name)

    mongo_client = MongoClient(**mongo_kwargs)
    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(
        client_config=mongo_config, db_name=mongo_db_name, locking_config=locking_config
    )

    mongo_client.drop_database(mongo_db_name)

    return MongoDocumentStore(store_config=store_config)


@pytest.fixture(scope="function", params=locking_scenarios)
def mongo_document_store(mongo_server_mock, request):
    locking_config_name = request.param
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()
    return mongo_document_store_fn(
        mongo_db_name=mongo_db_name,
        locking_config_name=locking_config_name,
        **mongo_kwargs,
    )


def mongo_queue_stash_fn(mongo_document_store):
    return QueueStash(store=mongo_document_store)


@pytest.fixture(scope="function", params=locking_scenarios)
def mongo_queue_stash(mongo_server_mock, request):
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()
    locking_config_name = request.param

    store = mongo_document_store_fn(
        mongo_db_name=mongo_db_name,
        locking_config_name=locking_config_name,
        **mongo_kwargs,
    )
    return mongo_queue_stash_fn(store)


def dict_store_partition_fn(
    locking_config_name: str = "nop",
):
    locking_config = str_to_locking_config(locking_config_name)
    store_config = DictStoreConfig(locking_config=locking_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return DictStorePartition(settings=settings, store_config=store_config)


@pytest.fixture(scope="function", params=locking_scenarios)
def dict_store_partition(request):
    locking_config_name = request.param
    return dict_store_partition_fn(locking_config_name=locking_config_name)


@pytest.fixture(scope="function", params=locking_scenarios)
def dict_action_store(request):
    locking_config_name = request.param
    locking_config = str_to_locking_config(locking_config_name)

    store_config = DictStoreConfig(locking_config=locking_config)
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return DictActionStore(store_config=store_config, root_verify_key=ver_key)


def dict_document_store_fn(locking_config_name: str = "nop"):
    locking_config = str_to_locking_config(locking_config_name)
    store_config = DictStoreConfig(locking_config=locking_config)
    return DictDocumentStore(store_config=store_config)


@pytest.fixture(scope="function", params=locking_scenarios)
def dict_document_store(request):
    locking_config_name = request.param
    return dict_document_store_fn(locking_config_name=locking_config_name)


def dict_queue_stash_fn(dict_document_store):
    return QueueStash(store=dict_document_store)


@pytest.fixture(scope="function")
def dict_queue_stash(dict_document_store):
    return dict_queue_stash_fn(dict_document_store)
