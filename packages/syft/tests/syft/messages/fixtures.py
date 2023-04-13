# third party
import pytest

# syft absolute
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.datetime import DateTime
from syft.core.node.new.linked_obj import LinkedObject
from syft.core.node.new.message_service import MessageService
from syft.core.node.new.message_stash import MessageStash
from syft.core.node.new.messages import CreateMessage
from syft.core.node.new.messages import Message
from syft.core.node.new.uid import UID
from syft.core.node.new.user import User
from syft.core.node.worker import Worker

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


@pytest.fixture(autouse=True)
def message_stash(document_store):
    return MessageStash(store=document_store)


@pytest.fixture(autouse=True)
def message_service(document_store):
    return MessageService(store=document_store)


@pytest.fixture(autouse=True)
def authed_context(admin_user: User, worker: Worker) -> AuthedServiceContext:
    return AuthedServiceContext(credentials=test_verify_key, node=worker)


@pytest.fixture(autouse=True)
def linked_object():
    return LinkedObject(
        node_uid=UID(),
        service_type=MessageService,
        object_type=Message,
        object_uid=UID(),
    )


@pytest.fixture(autouse=True)
def mock_create_message(faker) -> CreateMessage:
    test_signing_key1 = SyftSigningKey.generate()
    test_verify_key1 = test_signing_key1.verify_key
    test_signing_key2 = SyftSigningKey.generate()
    test_verify_key2 = test_signing_key2.verify_key

    mock_message = CreateMessage(
        subject="mock_created_message",
        id=UID(),
        node_uid=UID(),
        from_user_verify_key=test_verify_key1,
        to_user_verify_key=test_verify_key2,
        created_at=DateTime.now(),
    )

    return mock_message


@pytest.fixture(autouse=True)
def mock_message(
    message_stash: MessageStash,
) -> Message:
    mock_message = Message(
        subject="mock_message",
        node_uid=UID(),
        from_user_verify_key=SyftSigningKey.generate().verify_key,
        to_user_verify_key=SyftSigningKey.generate().verify_key,
        created_at=DateTime.now(),
    )

    result = message_stash.set(mock_message)
    assert result.is_ok()

    return mock_message
