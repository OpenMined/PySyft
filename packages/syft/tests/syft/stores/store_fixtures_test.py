# DO NOT IMPORT THIS FILE IN THE TEST SCRIPTS DIRECTLY.
#
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


@pytest.fixture(scope="function")
def sqlite_store_partition(sqlite_workspace: Tuple[Path, str]):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(settings=settings, store_config=store_config)

    res = store.init_store()
    assert res.is_ok()

    return store


@pytest.fixture(scope="function")
def sqlite_document_store(sqlite_workspace: Tuple[Path, str]):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    return SQLiteDocumentStore(store_config=store_config)


@pytest.fixture(scope="function")
def sqlite_queue_stash(sqlite_document_store: SQLiteDocumentStore):
    return QueueStash(store=sqlite_document_store)


@pytest.fixture(scope="function")
def sqlite_action_store(sqlite_workspace: Tuple[Path, str]):
    workspace, db_name = sqlite_workspace

    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return SQLiteActionStore(store_config=store_config, root_verify_key=ver_key)


@pytest.fixture(scope="function")
def mongo_store_partition(mongo_server_mock):
    mongo_db_name = generate_db_name()
    mongo_client = MongoClient(**mongo_server_mock.pmr_credentials.as_mongo_kwargs())

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=mongo_db_name)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    yield MongoStorePartition(settings=settings, store_config=store_config)

    mongo_client.drop_database(mongo_db_name)


@pytest.fixture(scope="function")
def mongo_document_store(mongo_server_mock):
    mongo_db_name = generate_db_name()
    mongo_client = MongoClient(**mongo_server_mock.pmr_credentials.as_mongo_kwargs())

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=mongo_db_name)

    mongo_client.drop_database(mongo_db_name)

    return MongoDocumentStore(store_config=store_config)


@pytest.fixture(scope="function")
def mongo_queue_stash(mongo_document_store):
    return QueueStash(store=mongo_document_store)


@pytest.fixture(scope="function")
def dict_store_partition():
    store_config = DictStoreConfig()
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return DictStorePartition(settings=settings, store_config=store_config)


@pytest.fixture(scope="function")
def dict_action_store():
    store_config = DictStoreConfig()
    ver_key = SyftVerifyKey.from_string(test_verify_key_string_root)
    return DictActionStore(store_config=store_config, root_verify_key=ver_key)


@pytest.fixture(scope="function")
def dict_document_store():
    store_config = DictStoreConfig()
    return DictDocumentStore(store_config=store_config)


@pytest.fixture(scope="function")
def dict_queue_stash(dict_document_store):
    return QueueStash(store=dict_document_store)
