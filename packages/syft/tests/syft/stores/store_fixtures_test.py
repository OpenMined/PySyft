# stdlib
from pathlib import Path
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
from .store_constants_test import test_verify_key_string_root
from .store_constants_test import workspace
from .store_mocks_test import MockObjectType

mongo_server_mock = create_mongo_fixture(scope="session")


@pytest.fixture(scope="function")
def sqlite_workspace() -> Generator:
    sqlite_db_name = generate_db_name()

    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / sqlite_db_name

    if db_path.exists():
        db_path.unlink()

    yield workspace, sqlite_db_name

    if db_path.exists():
        db_path.unlink()


def sqlite_store_partition_fn(sqlite_workspace: Tuple[Path, str]):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(settings=settings, store_config=store_config)

    res = store.init_store()
    assert res.is_ok()

    return store


@pytest.fixture(scope="function")
def sqlite_store_partition(sqlite_workspace: Tuple[Path, str]):
    return sqlite_store_partition_fn(sqlite_workspace)


def sqlite_document_store_fn(sqlite_workspace: Tuple[Path, str]):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    return SQLiteDocumentStore(store_config=store_config)


@pytest.fixture(scope="function")
def sqlite_document_store(sqlite_workspace: Tuple[Path, str]):
    return sqlite_document_store_fn(sqlite_workspace)


def sqlite_queue_stash_fn(sqlite_workspace: Tuple[Path, str]):
    store = sqlite_document_store_fn(sqlite_workspace)
    return QueueStash(store=store)


@pytest.fixture(scope="function")
def sqlite_queue_stash(sqlite_workspace: Tuple[Path, str]):
    return sqlite_queue_stash_fn(sqlite_workspace)


@pytest.fixture(scope="function")
def sqlite_action_store(sqlite_workspace: Tuple[Path, str]):
    workspace, db_name = sqlite_workspace

    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return SQLiteActionStore(store_config=store_config, root_verify_key=ver_key)


def mongo_store_partition_fn(mongo_db_name: str = "mongo_db", **mongo_kwargs):
    mongo_client = MongoClient(**mongo_kwargs)

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=mongo_db_name)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return MongoStorePartition(settings=settings, store_config=store_config)


@pytest.fixture(scope="function")
def mongo_store_partition(mongo_server_mock):
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    yield mongo_store_partition_fn(mongo_db_name=mongo_db_name, **mongo_kwargs)

    # cleanup db
    mongo_client = MongoClient(**mongo_kwargs)
    mongo_client.drop_database(mongo_db_name)


def mongo_document_store_fn(mongo_db_name: str = "mongo_db", **mongo_kwargs):
    mongo_client = MongoClient(**mongo_kwargs)

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=mongo_db_name)

    mongo_client.drop_database(mongo_db_name)

    return MongoDocumentStore(store_config=store_config)


@pytest.fixture(scope="function")
def mongo_document_store(mongo_server_mock):
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()
    return mongo_document_store_fn(mongo_db_name=mongo_db_name, **mongo_kwargs)


def mongo_queue_stash_fn(mongo_document_store):
    return QueueStash(store=mongo_document_store)


@pytest.fixture(scope="function")
def mongo_queue_stash(mongo_server_mock):
    mongo_db_name = generate_db_name()
    mongo_kwargs = mongo_server_mock.pmr_credentials.as_mongo_kwargs()

    store = mongo_document_store_fn(mongo_db_name=mongo_db_name, **mongo_kwargs)
    return mongo_queue_stash_fn(store)


def dict_store_partition_fn():
    store_config = DictStoreConfig()
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return DictStorePartition(settings=settings, store_config=store_config)


@pytest.fixture(scope="function")
def dict_store_partition():
    return dict_store_partition_fn()


@pytest.fixture(scope="function")
def dict_action_store():
    store_config = DictStoreConfig()
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return DictActionStore(store_config=store_config, root_verify_key=ver_key)


def dict_document_store_fn():
    store_config = DictStoreConfig()
    return DictDocumentStore(store_config=store_config)


@pytest.fixture(scope="function")
def dict_document_store():
    return dict_document_store_fn()


def dict_queue_stash_fn(dict_document_store):
    return QueueStash(store=dict_document_store)


@pytest.fixture(scope="function")
def dict_queue_stash(dict_document_store):
    return dict_queue_stash_fn(dict_document_store)
