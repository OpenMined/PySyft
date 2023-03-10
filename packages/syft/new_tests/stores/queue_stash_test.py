# stdlib
from pathlib import Path
from typing import Any
from typing import Generator

# third party
from pymongo import MongoClient
import pytest
from pytest_mock_resources import create_mongo_fixture

# syft absolute
from syft.core.node.new.dict_document_store import DictDocumentStore
from syft.core.node.new.dict_document_store import DictStoreConfig
from syft.core.node.new.document_store import PartitionSettings
from syft.core.node.new.mongo_client import MongoStoreClientConfig
from syft.core.node.new.mongo_document_store import MongoDocumentStore
from syft.core.node.new.mongo_document_store import MongoStoreConfig
from syft.core.node.new.queue_stash import QueueStash
from syft.core.node.new.sqlite_document_store import SQLiteDocumentStore
from syft.core.node.new.sqlite_document_store import SQLiteStoreClientConfig
from syft.core.node.new.sqlite_document_store import SQLiteStoreConfig

# relative
from .store_mocks import MockObjectType

mongo = create_mongo_fixture(scope="session")
workspace = Path("workspace")
db_name = "testing"


@pytest.fixture(autouse=True)
def prepare_workspace() -> Generator:
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / db_name

    if db_path.exists():
        db_path.unlink()

    yield

    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def dict_store():
    store_config = DictStoreConfig()
    return DictDocumentStore(store_config=store_config)


@pytest.fixture
def sqlite_store():
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)
    store_config = SQLiteStoreConfig(client_config=sqlite_config)
    return SQLiteDocumentStore(store_config=store_config)


@pytest.fixture
def mongo_store(mongo):
    mongo_client = MongoClient(**mongo.pmr_credentials.as_mongo_kwargs())

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(client_config=mongo_config, db_name=db_name)
    PartitionSettings(name="test", object_type=MockObjectType)

    mongo_client.drop_database(db_name)

    return MongoDocumentStore(store_config=store_config)


@pytest.mark.parametrize(
    "store",
    [
        pytest.lazy_fixture("dict_store"),
        pytest.lazy_fixture("sqlite_store"),
        pytest.lazy_fixture("mongo_store"),
    ],
)
def test_queue_stash_sanity(store: Any) -> None:
    queue = QueueStash(store=store)

    assert len(queue) == 0
