# stdlib
import uuid

# syft absolute
from syft.server.credentials import SyftVerifyKey
from syft.service.action.action_permissions import ActionObjectPermission
from syft.service.action.action_permissions import ActionPermission
from syft.service.user.user import User
from syft.service.user.user import UserCreate
from syft.service.user.user_roles import ServiceRole
from syft.service.user.user_stash import UserStash
from syft.store.db.sqlite import SQLiteDBConfig
from syft.store.db.sqlite import SQLiteDBManager
from syft.store.document_store import DocumentStore
from syft.types.uid import UID

# relative
from .store_constants_test import TEST_SIGNING_KEY_NEW_ADMIN
from .store_constants_test import TEST_VERIFY_KEY_NEW_ADMIN


def document_store_with_admin(
    server_uid: UID, verify_key: SyftVerifyKey
) -> DocumentStore:
    config = SQLiteDBConfig()
    document_store = SQLiteDBManager(
        server_uid=server_uid, root_verify_key=verify_key, config=config
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
        obj=admin_user,
        add_permissions=[
            ActionObjectPermission(
                uid=admin_user.id, permission=ActionPermission.ALL_READ
            ),
        ],
    )

    return document_store
