# third party
import pytest

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.node.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.notification.notification_service import NotificationService
from syft.service.notification.notification_stash import NotificationStash
from syft.service.notification.notifications import Createnotification
from syft.service.notification.notifications import Notification
from syft.service.user.user import User
from syft.store.linked_obj import LinkedObject
from syft.types.datetime import DateTime
from syft.types.uid import UID

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


@pytest.fixture(autouse=True)
def notification_stash(document_store):
    return NotificationStash(store=document_store)


@pytest.fixture(autouse=True)
def notification_service(document_store):
    return NotificationService(store=document_store)


@pytest.fixture(autouse=True)
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    return AuthedServiceContext(credentials=test_verify_key, node=worker)


@pytest.fixture(autouse=True)
def linked_object():
    return LinkedObject(
        node_uid=UID(),
        service_type=NotificationService,
        object_type=Notification,
        object_uid=UID(),
    )


@pytest.fixture(autouse=True)
def mock_create_notification(faker) -> Createnotification:
    test_signing_key1 = SyftSigningKey.generate()
    test_verify_key1 = test_signing_key1.verify_key
    test_signing_key2 = SyftSigningKey.generate()
    test_verify_key2 = test_signing_key2.verify_key

    mock_notification = Createnotification(
        subject="mock_created_notification",
        id=UID(),
        node_uid=UID(),
        from_user_verify_key=test_verify_key1,
        to_user_verify_key=test_verify_key2,
        created_at=DateTime.now(),
    )

    return mock_notification


@pytest.fixture(autouse=True)
def mock_notification(
    root_verify_key,
    notification_stash: NotificationStash,
) -> Notification:
    mock_notification = Notification(
        subject="mock_notification",
        node_uid=UID(),
        from_user_verify_key=SyftSigningKey.generate().verify_key,
        to_user_verify_key=SyftSigningKey.generate().verify_key,
        created_at=DateTime.now(),
    )

    result = notification_stash.set(root_verify_key, mock_notification)
    assert result.is_ok()

    return mock_notification
