# third party
import pytest

# syft absolute
from syft.serde.serializable import serializable
from syft.server.credentials import SyftSigningKey
from syft.server.credentials import SyftVerifyKey
from syft.server.worker import Worker
from syft.service.context import AuthedServiceContext
from syft.service.notification.email_templates import EmailTemplate
from syft.service.notification.notification_service import NotificationService
from syft.service.notification.notification_stash import NotificationStash
from syft.service.notification.notifications import CreateNotification
from syft.service.notification.notifications import Notification
from syft.service.notifier.notifier_enums import NOTIFIERS
from syft.service.user.user import User
from syft.store.linked_obj import LinkedObject
from syft.types.datetime import DateTime
from syft.types.uid import UID

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


@pytest.fixture
def notification_stash(document_store):
    yield NotificationStash(store=document_store)


@pytest.fixture
def notification_service(document_store):
    yield NotificationService(store=document_store)


@pytest.fixture
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    yield AuthedServiceContext(credentials=test_verify_key, server=worker)


@pytest.fixture
def linked_object():
    yield LinkedObject(
        server_uid=UID(),
        service_type=NotificationService,
        object_type=Notification,
        object_uid=UID(),
    )


@serializable(canonical_name="NewEmail", version=1)
class NewEmail(EmailTemplate):
    @staticmethod
    def email_title(notification: "Notification", context) -> str:
        return f"Welcome to {context.server.name} server!"

    @staticmethod
    def email_body(notification: "Notification", context) -> str:
        return "x"


@pytest.fixture
def mock_create_notification(faker) -> CreateNotification:
    test_signing_key1 = SyftSigningKey.generate()
    test_verify_key1 = test_signing_key1.verify_key
    test_signing_key2 = SyftSigningKey.generate()
    test_verify_key2 = test_signing_key2.verify_key

    mock_notification = CreateNotification(
        subject="mock_created_notification",
        id=UID(),
        server_uid=UID(),
        notifier_types=[NOTIFIERS.EMAIL],
        from_user_verify_key=test_verify_key1,
        to_user_verify_key=test_verify_key2,
        created_at=DateTime.now(),
        email_template=NewEmail,
    )

    yield mock_notification


@pytest.fixture
def mock_notification(
    root_verify_key,
    notification_stash: NotificationStash,
) -> Notification:
    mock_notification = Notification(
        subject="mock_notification",
        server_uid=UID(),
        from_user_verify_key=SyftSigningKey.generate().verify_key,
        to_user_verify_key=SyftSigningKey.generate().verify_key,
        created_at=DateTime.now(),
    )

    result = notification_stash.set(root_verify_key, mock_notification)
    assert result.is_ok()

    yield mock_notification
