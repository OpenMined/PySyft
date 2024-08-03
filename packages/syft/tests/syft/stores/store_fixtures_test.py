# stdlib
import contextlib
import os
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from secrets import token_hex

# third party
import pytest

# syft absolute
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_permissions import (
    ActionObjectPermission,
    ActionPermission,
)
from syft.service.action.action_store import (
    DictActionStore,
    MongoActionStore,
    SQLiteActionStore,
)
from syft.service.queue.queue_stash import QueueStash
from syft.service.user.user import User, UserCreate
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_stash import UserStash
from syft.store.dict_document_store import (
    DictDocumentStore,
    DictStoreConfig,
    DictStorePartition,
)
from syft.store.document_store import DocumentStore, PartitionSettings
from syft.store.locks import LockingConfig, NoLockingConfig, ThreadingLockingConfig
from syft.store.mongo_client import MongoStoreClientConfig
from syft.store.mongo_document_store import (
    MongoDocumentStore,
    MongoStoreConfig,
    MongoStorePartition,
)
from syft.store.sqlite_document_store import (
    SQLiteDocumentStore,
    SQLiteStoreClientConfig,
    SQLiteStoreConfig,
    SQLiteStorePartition,
)
from syft.types.uid import UID

# relative
from .store_constants_test import (
    TEST_SIGNING_KEY_NEW_ADMIN,
    TEST_VERIFY_KEY_NEW_ADMIN,
    TEST_VERIFY_KEY_STRING_ROOT,
)
from .store_mocks_test import MockObjectType

MONGO_CLIENT_CACHE = None

locking_scenarios = [
    "nop",
    "threading",
]


def str_to_locking_config(conf: str) -> LockingConfig:
    if conf == "nop":
        return NoLockingConfig()
    elif conf == "threading":
        return ThreadingLockingConfig()
    else:
        msg = f"unknown locking config {conf}"
        raise NotImplementedError(msg)


def document_store_with_admin(
    server_uid: UID, verify_key: SyftVerifyKey,
) -> DocumentStore:
    document_store = DictDocumentStore(
        server_uid=server_uid, root_verify_key=verify_key,
    )

    password = uuid.uuid4().hex

    user_stash = UserStash(store=document_store)
    admin_user = UserCreate(
        email="mail@example.org",
        name="Admin",
        password=password,
        password_verify=password,
        role=ServiceRole.ADMIN,
    ).to(User)

    admin_user.signing_key = TEST_SIGNING_KEY_NEW_ADMIN
    admin_user.verify_key = TEST_VERIFY_KEY_NEW_ADMIN

    user_stash.set(
        credentials=verify_key,
        user=admin_user,
        add_permissions=[
            ActionObjectPermission(
                uid=admin_user.id, permission=ActionPermission.ALL_READ,
            ),
        ],
    )

    return document_store


@pytest.fixture()
def sqlite_workspace() -> Generator:
    sqlite_db_name = token_hex(8) + ".sqlite"
    root = os.getenv("SYFT_TEMP_ROOT", "syft")
    sqlite_workspace_folder = Path(
        tempfile.gettempdir(), root, "fixture_sqlite_workspace",
    )
    sqlite_workspace_folder.mkdir(parents=True, exist_ok=True)

    db_path = sqlite_workspace_folder / sqlite_db_name

    if db_path.exists():
        db_path.unlink()

    yield sqlite_workspace_folder, sqlite_db_name

    with contextlib.suppress(BaseException):
        db_path.exists() and db_path.unlink()


def sqlite_store_partition_fn(
    root_verify_key,
    sqlite_workspace: tuple[Path, str],
    locking_config_name: str = "nop",
):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)

    locking_config = str_to_locking_config(locking_config_name)
    store_config = SQLiteStoreConfig(
        client_config=sqlite_config, locking_config=locking_config,
    )

    settings = PartitionSettings(name="test", object_type=MockObjectType)

    store = SQLiteStorePartition(
        UID(), root_verify_key, settings=settings, store_config=store_config,
    )

    res = store.init_store()
    assert res.is_ok()

    return store


@pytest.fixture(params=locking_scenarios)
def sqlite_store_partition(
    root_verify_key, sqlite_workspace: tuple[Path, str], request,
):
    locking_config_name = request.param
    return sqlite_store_partition_fn(
        root_verify_key, sqlite_workspace, locking_config_name=locking_config_name,
    )



def sqlite_document_store_fn(
    root_verify_key,
    sqlite_workspace: tuple[Path, str],
    locking_config_name: str = "nop",
):
    workspace, db_name = sqlite_workspace
    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)

    locking_config = str_to_locking_config(locking_config_name)
    store_config = SQLiteStoreConfig(
        client_config=sqlite_config, locking_config=locking_config,
    )

    return SQLiteDocumentStore(UID(), root_verify_key, store_config=store_config)


@pytest.fixture(params=locking_scenarios)
def sqlite_document_store(root_verify_key, sqlite_workspace: tuple[Path, str], request):
    locking_config_name = request.param
    return sqlite_document_store_fn(
        root_verify_key, sqlite_workspace, locking_config_name=locking_config_name,
    )


def sqlite_queue_stash_fn(
    root_verify_key,
    sqlite_workspace: tuple[Path, str],
    locking_config_name: str = "threading",
):
    store = sqlite_document_store_fn(
        root_verify_key,
        sqlite_workspace,
        locking_config_name=locking_config_name,
    )
    return QueueStash(store=store)


@pytest.fixture(params=locking_scenarios)
def sqlite_queue_stash(root_verify_key, sqlite_workspace: tuple[Path, str], request):
    locking_config_name = request.param
    return sqlite_queue_stash_fn(
        root_verify_key, sqlite_workspace, locking_config_name=locking_config_name,
    )


@pytest.fixture(params=locking_scenarios)
def sqlite_action_store(sqlite_workspace: tuple[Path, str], request):
    workspace, db_name = sqlite_workspace
    locking_config_name = request.param

    sqlite_config = SQLiteStoreClientConfig(filename=db_name, path=workspace)

    locking_config = str_to_locking_config(locking_config_name)
    store_config = SQLiteStoreConfig(
        client_config=sqlite_config,
        locking_config=locking_config,
    )

    ver_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_ROOT)

    server_uid = UID()
    document_store = document_store_with_admin(server_uid, ver_key)

    return SQLiteActionStore(
        server_uid=server_uid,
        store_config=store_config,
        root_verify_key=ver_key,
        document_store=document_store,
    )


def mongo_store_partition_fn(
    mongo_client,
    root_verify_key,
    mongo_db_name: str = "mongo_db",
    locking_config_name: str = "nop",
):
    mongo_config = MongoStoreClientConfig(client=mongo_client)

    locking_config = str_to_locking_config(locking_config_name)

    store_config = MongoStoreConfig(
        client_config=mongo_config,
        db_name=mongo_db_name,
        locking_config=locking_config,
    )
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return MongoStorePartition(
        UID(), root_verify_key, settings=settings, store_config=store_config,
    )


@pytest.fixture(params=locking_scenarios)
def mongo_store_partition(root_verify_key, mongo_client, request):
    mongo_db_name = token_hex(8)
    locking_config_name = request.param

    partition = mongo_store_partition_fn(
        mongo_client,
        root_verify_key,
        mongo_db_name=mongo_db_name,
        locking_config_name=locking_config_name,
    )
    yield partition

    # cleanup db
    with contextlib.suppress(BaseException):
        mongo_client.drop_database(mongo_db_name)


def mongo_document_store_fn(
    mongo_client,
    root_verify_key,
    mongo_db_name: str = "mongo_db",
    locking_config_name: str = "nop",
):
    locking_config = str_to_locking_config(locking_config_name)
    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(
        client_config=mongo_config, db_name=mongo_db_name, locking_config=locking_config,
    )

    mongo_client.drop_database(mongo_db_name)

    return MongoDocumentStore(UID(), root_verify_key, store_config=store_config)


@pytest.fixture(params=locking_scenarios)
def mongo_document_store(root_verify_key, mongo_client, request):
    locking_config_name = request.param
    mongo_db_name = token_hex(8)
    return mongo_document_store_fn(
        mongo_client,
        root_verify_key,
        mongo_db_name=mongo_db_name,
        locking_config_name=locking_config_name,
    )


def mongo_queue_stash_fn(mongo_document_store):
    return QueueStash(store=mongo_document_store)


@pytest.fixture(params=locking_scenarios)
def mongo_queue_stash(root_verify_key, mongo_client, request):
    mongo_db_name = token_hex(8)
    locking_config_name = request.param

    store = mongo_document_store_fn(
        mongo_client,
        root_verify_key,
        mongo_db_name=mongo_db_name,
        locking_config_name=locking_config_name,
    )
    return mongo_queue_stash_fn(store)


@pytest.fixture(params=locking_scenarios)
def mongo_action_store(mongo_client, request):
    mongo_db_name = token_hex(8)
    locking_config_name = request.param
    locking_config = str_to_locking_config(locking_config_name)

    mongo_config = MongoStoreClientConfig(client=mongo_client)
    store_config = MongoStoreConfig(
        client_config=mongo_config, db_name=mongo_db_name, locking_config=locking_config,
    )
    ver_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_ROOT)
    server_uid = UID()
    document_store = document_store_with_admin(server_uid, ver_key)
    return MongoActionStore(
        server_uid=server_uid,
        store_config=store_config,
        root_verify_key=ver_key,
        document_store=document_store,
    )



def dict_store_partition_fn(
    root_verify_key,
    locking_config_name: str = "nop",
):
    locking_config = str_to_locking_config(locking_config_name)
    store_config = DictStoreConfig(locking_config=locking_config)
    settings = PartitionSettings(name="test", object_type=MockObjectType)

    return DictStorePartition(
        UID(), root_verify_key, settings=settings, store_config=store_config,
    )


@pytest.fixture(params=locking_scenarios)
def dict_store_partition(root_verify_key, request):
    locking_config_name = request.param
    return dict_store_partition_fn(
        root_verify_key, locking_config_name=locking_config_name,
    )


@pytest.fixture(params=locking_scenarios)
def dict_action_store(request):
    locking_config_name = request.param
    locking_config = str_to_locking_config(locking_config_name)

    store_config = DictStoreConfig(locking_config=locking_config)
    ver_key = SyftVerifyKey.from_string(TEST_VERIFY_KEY_STRING_ROOT)
    server_uid = UID()
    document_store = document_store_with_admin(server_uid, ver_key)

    return DictActionStore(
        server_uid=server_uid,
        store_config=store_config,
        root_verify_key=ver_key,
        document_store=document_store,
    )


def dict_document_store_fn(root_verify_key, locking_config_name: str = "nop"):
    locking_config = str_to_locking_config(locking_config_name)
    store_config = DictStoreConfig(locking_config=locking_config)
    return DictDocumentStore(UID(), root_verify_key, store_config=store_config)


@pytest.fixture(params=locking_scenarios)
def dict_document_store(root_verify_key, request):
    locking_config_name = request.param
    return dict_document_store_fn(
        root_verify_key, locking_config_name=locking_config_name,
    )


def dict_queue_stash_fn(dict_document_store):
    return QueueStash(store=dict_document_store)


@pytest.fixture()
def dict_queue_stash(dict_document_store):
    return dict_queue_stash_fn(dict_document_store)
