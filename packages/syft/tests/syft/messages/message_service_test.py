# third party
from pytest import MonkeyPatch
from result import Err
from result import Ok

# syft absolute
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.datetime import DateTime
from syft.core.node.new.document_store import DocumentStore
from syft.core.node.new.linked_obj import LinkedObject
from syft.core.node.new.message_service import MessageService
from syft.core.node.new.message_stash import MessageStash
from syft.core.node.new.messages import CreateMessage
from syft.core.node.new.messages import Message
from syft.core.node.new.messages import MessageStatus
from syft.core.node.new.response import SyftError
from syft.core.node.new.response import SyftSuccess
from syft.core.node.new.uid import UID

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


def add_mock_message(
    root_verify_key,
    message_stash: MessageStash,
    from_user_verify_key: SyftVerifyKey,
    to_user_verify_key: SyftVerifyKey,
) -> Message:
    # prepare: add mock message

    message_status_undelivered = MessageStatus(0)

    mock_message = Message(
        subject="mock_message",
        node_uid=UID(),
        from_user_verify_key=from_user_verify_key,
        to_user_verify_key=to_user_verify_key,
        created_at=DateTime.now(),
        status=message_status_undelivered,
    )

    result = message_stash.set(root_verify_key, mock_message)
    assert result.is_ok()

    return mock_message


def test_messageservice_send_success(
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    mock_create_message: CreateMessage,
    document_store,
) -> None:
    test_message_service = MessageService(store=document_store)

    expected_message = mock_create_message.to(Message, authed_context)

    def mock_set(message_service: MessageService) -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(message_service.stash, "set", mock_set)
    response = test_message_service.send(authed_context, mock_create_message)

    assert isinstance(response, Message)


def test_messageservice_send_error_on_set(
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    mock_create_message: CreateMessage,
) -> None:
    def mock_set(credentials: SyftVerifyKey, message_service: MessageService) -> Err:
        return Err(expected_error)

    test_message_service = message_service
    expected_error = "Failed to set message."

    monkeypatch.setattr(message_service.stash, "set", mock_set)
    response = test_message_service.send(authed_context, mock_create_message)

    assert isinstance(response, SyftError)
    assert response.message == expected_error


def test_messageservice_get_all_success(
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_message_service = MessageService(document_store)
    test_stash = MessageStash(store=document_store)

    expected_message = add_mock_message(
        authed_context.credentials, test_stash, random_verify_key, test_verify_key
    )

    def mock_get_all_inbox_for_verify_key() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        message_service.stash,
        "get_all_inbox_for_verify_key",
        mock_get_all_inbox_for_verify_key,
    )

    response = test_message_service.get_all(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Message)
    assert response[0] == expected_message


def test_messageservice_get_all_error_on_get_all_inbox(
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all inbox."

    def mock_get_all_inbox_for_verify_key(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        message_service.stash,
        "get_all_inbox_for_verify_key",
        mock_get_all_inbox_for_verify_key,
    )

    response = message_service.get_all(authed_context)

    assert isinstance(response, SyftError)
    assert response.message == expected_error


def test_messageservice_get_sent_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_message_service = MessageService(document_store)
    test_stash = MessageStash(store=document_store)

    expected_message = add_mock_message(
        authed_context.credentials, test_stash, test_verify_key, random_verify_key
    )

    def mock_get_all_sent_for_verify_key(credentials, verify_key) -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        message_service.stash,
        "get_all_sent_for_verify_key",
        mock_get_all_sent_for_verify_key,
    )

    response = test_message_service.get_all_sent(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Message)
    assert response[0] == expected_message


def test_messageservice_get_all_error_on_get_all_sent(
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all sent."

    def mock_get_all_sent_for_verify_key(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        message_service.stash,
        "get_all_sent_for_verify_key",
        mock_get_all_sent_for_verify_key,
    )

    response = message_service.get_all_sent(authed_context)

    assert isinstance(response, SyftError)
    assert response.message == expected_error


def test_messageservice_get_all_by_verify_key_for_status_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_message_service = MessageService(document_store)
    test_stash = MessageStash(store=document_store)
    messeage_status_undelivered = MessageStatus(0)

    expected_message = add_mock_message(
        authed_context.credentials, test_stash, random_verify_key, test_verify_key
    )

    def mock_get_all_by_verify_key_for_status() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        message_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = test_message_service.get_all_for_status(
        authed_context, messeage_status_undelivered
    )

    assert len(response) == 1
    assert isinstance(response[0], Message)
    assert response[0] == expected_message


def test_messageservice_error_on_get_all_for_status(
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
) -> None:
    messeage_status_undelivered = MessageStatus(0)
    expected_error = "Failed to get all for status."

    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey, status: MessageStatus
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        message_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = message_service.get_all_for_status(
        authed_context,
        messeage_status_undelivered,
    )

    assert isinstance(response, SyftError)
    assert response.message == expected_error


def test_messageservice_mark_as_deilvered_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_message_service = MessageService(document_store)
    test_stash = MessageStash(store=document_store)
    messeage_status_undelivered = MessageStatus(0)
    messeage_status_delivered = MessageStatus(1)

    expected_message = add_mock_message(
        authed_context.credentials, test_stash, test_verify_key, random_verify_key
    )

    assert expected_message.status == messeage_status_undelivered

    def mock_update_message_status() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        message_service.stash,
        "update_message_status",
        mock_update_message_status,
    )

    response = test_message_service.mark_as_delivered(
        authed_context, expected_message.id
    )

    assert response.status == messeage_status_delivered


def test_messageservice_mark_as_delivered_error_on_update_message_status(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = MessageStash(store=document_store)

    expected_message = add_mock_message(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )
    expected_error = "Failed to update message status."

    def mock_update_message_status(
        credentials: SyftVerifyKey, uid: UID, status: MessageStatus
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        message_service.stash,
        "update_message_status",
        mock_update_message_status,
    )

    response = message_service.mark_as_delivered(authed_context, expected_message.id)

    assert isinstance(response, SyftError)
    assert response.message == expected_error


# TODO: Fix this test - unsure how to return a LinkedObject Message.
# Test executes code but does not return a Message object.
def test_messageservice_resolve_object_success(
    monkeypatch: MonkeyPatch,
    authed_context: AuthedServiceContext,
    linked_object: LinkedObject,
    document_store: DocumentStore,
) -> None:
    test_message_service = MessageService(document_store)

    def mock_get_service(linked_obj: LinkedObject) -> MessageService:
        return test_message_service

    monkeypatch.setattr(
        authed_context.node,
        "get_service",
        mock_get_service,
    )

    def mock_resolve_link(
        context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> Ok:
        return Ok(None)

    monkeypatch.setattr(
        test_message_service,
        "resolve_link",
        mock_resolve_link,
    )

    response = test_message_service.resolve_object(authed_context, linked_object)

    assert response is None


def test_messageservice_resolve_object_error_on_resolve_link(
    monkeypatch: MonkeyPatch,
    authed_context: AuthedServiceContext,
    linked_object: LinkedObject,
    document_store: DocumentStore,
) -> None:
    test_message_service = MessageService(document_store)
    expected_error = "Failed to resolve link."

    def mock_get_service(linked_obj: LinkedObject) -> MessageService:
        return test_message_service

    monkeypatch.setattr(
        authed_context.node,
        "get_service",
        mock_get_service,
    )

    def mock_resolve_link(
        context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        test_message_service,
        "resolve_link",
        mock_resolve_link,
    )

    response = test_message_service.resolve_object(authed_context, linked_object)

    assert isinstance(response, SyftError)
    assert response.message == expected_error


def test_messageservice_clear_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_message_service = MessageService(document_store)
    test_stash = MessageStash(store=document_store)

    expected_success_message = "All messages cleared !!"
    add_mock_message(
        authed_context.credentials, test_stash, random_verify_key, test_verify_key
    )
    inbox_before_delete = test_message_service.get_all(authed_context)

    assert len(inbox_before_delete) == 1

    def mock_delete_all_for_verify_key(credentials, verify_key) -> Ok:
        return Ok(SyftSuccess.message)

    monkeypatch.setattr(
        message_service.stash,
        "delete_all_for_verify_key",
        mock_delete_all_for_verify_key,
    )

    response = test_message_service.clear(authed_context)
    inbox_after_delete = test_message_service.get_all(authed_context)

    assert response.message == expected_success_message
    assert len(inbox_after_delete) == 0


def test_messageservice_clear_error_on_delete_all_for_verify_key(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    message_service: MessageService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_message_service = MessageService(document_store)
    test_stash = MessageStash(store=document_store)

    expected_error = "Failed to clear messages."
    add_mock_message(
        authed_context.credentials, test_stash, random_verify_key, test_verify_key
    )
    inbox_before_delete = test_message_service.get_all(authed_context)

    assert len(inbox_before_delete) == 1

    def mock_delete_all_for_verify_key(**kwargs) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        test_message_service.stash,
        "delete_all_for_verify_key",
        mock_delete_all_for_verify_key,
    )

    response = test_message_service.clear(authed_context)
    inbox_after_delete = test_message_service.get_all(authed_context)

    assert isinstance(response, SyftError)
    assert response.message == expected_error
    assert len(inbox_after_delete) == 1
