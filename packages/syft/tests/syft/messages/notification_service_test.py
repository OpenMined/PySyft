# third party
from pytest import MonkeyPatch
from result import Err
from result import Ok

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.service.context import AuthedServiceContext
from syft.service.notification.notification_service import NotificationService
from syft.service.notification.notification_stash import NotificationStash
from syft.service.notification.notifications import CreateNotification
from syft.service.notification.notifications import Notification
from syft.service.notification.notifications import NotificationStatus
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.store.document_store import DocumentStore
from syft.store.linked_obj import LinkedObject
from syft.types.datetime import DateTime
from syft.types.uid import UID

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


def add_mock_notification(
    root_verify_key,
    notification_stash: NotificationStash,
    from_user_verify_key: SyftVerifyKey,
    to_user_verify_key: SyftVerifyKey,
    status: NotificationStatus,
) -> Notification:
    # prepare: add mock

    mock_notification = Notification(
        subject="mock_notification",
        node_uid=UID(),
        from_user_verify_key=from_user_verify_key,
        to_user_verify_key=to_user_verify_key,
        created_at=DateTime.now(),
        status=status,
    )

    result = notification_stash.set(root_verify_key, mock_notification)
    assert result.is_ok()

    return mock_notification


def test_service_send_success(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    mock_create_notification: CreateNotification,
    document_store,
) -> None:
    test_notification_service = NotificationService(store=document_store)

    expected_message = mock_create_notification.to(Notification, authed_context)

    def mock_set(notification_service: NotificationService) -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(notification_service.stash, "set", mock_set)
    response = test_notification_service.send(authed_context, mock_create_notification)

    assert isinstance(response, Notification)


def test_service_send_error_on_set(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    mock_create_notification: CreateNotification,
) -> None:
    def mock_set(credentials: SyftVerifyKey, _service: NotificationService) -> Err:
        return Err(expected_error)

    test_notification_service = notification_service
    expected_error = "Failed to set ."

    monkeypatch.setattr(notification_service.stash, "set", mock_set)
    response = test_notification_service.send(authed_context, mock_create_notification)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_get_all_success(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_message = add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )

    def mock_get_all_inbox_for_verify_key() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_inbox_for_verify_key",
        mock_get_all_inbox_for_verify_key,
    )

    response = test_notification_service.get_all(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    assert response[0] == expected_message


def test_service_get_all_error_on_get_all_inbox(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all inbox."

    def mock_get_all_inbox_for_verify_key(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_inbox_for_verify_key",
        mock_get_all_inbox_for_verify_key,
    )

    response = notification_service.get_all(authed_context)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_get_sent_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_message = add_mock_notification(
        authed_context.credentials,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.UNREAD,
    )

    def mock_get_all_sent_for_verify_key(credentials, verify_key) -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_sent_for_verify_key",
        mock_get_all_sent_for_verify_key,
    )

    response = test_notification_service.get_all_sent(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    assert response[0] == expected_message


def test_service_get_all_error_on_get_all_sent(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all sent."

    def mock_get_all_sent_for_verify_key(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_sent_for_verify_key",
        mock_get_all_sent_for_verify_key,
    )

    response = notification_service.get_all_sent(authed_context)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_get_all_for_status_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_message = add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )

    def mock_get_all_by_verify_key_for_status() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = test_notification_service.get_all_for_status(
        authed_context, NotificationStatus.UNREAD
    )

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    assert response[0] == expected_message


def test_service_error_on_get_all_for_status(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all for status."

    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = notification_service.get_all_for_status(
        authed_context,
        NotificationStatus.UNREAD,
    )

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_get_all_read_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_message = add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.READ,
    )

    def mock_get_all_by_verify_key_for_status() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = test_notification_service.get_all_read(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    assert response[0] == expected_message


def test_service_error_on_get_all_read(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all for status."

    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = notification_service.get_all_read(authed_context)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_get_all_unread_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_message = add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )

    def mock_get_all_by_verify_key_for_status() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = test_notification_service.get_all_unread(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    assert response[0] == expected_message


def test_service_error_on_get_all_unread(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all for status."

    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = notification_service.get_all_unread(authed_context)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_mark_as_read_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_message = add_mock_notification(
        authed_context.credentials,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.UNREAD,
    )

    assert expected_message.status == NotificationStatus.UNREAD

    def mock_update_notification_status() -> Ok:
        return Ok(expected_message)

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    response = test_notification_service.mark_as_read(
        authed_context, expected_message.id
    )

    assert response.status == NotificationStatus.READ


def test_service_mark_as_read_error_on_update__status(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    expected_ = add_mock_notification(
        root_verify_key,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.UNREAD,
    )
    expected_error = "Failed to update  status."

    def mock_update_notification_status(
        credentials: SyftVerifyKey, uid: UID, status: NotificationStatus
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    response = notification_service.mark_as_read(authed_context, expected_.id)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_mark_as_unread_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_ = add_mock_notification(
        authed_context.credentials,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.READ,
    )

    assert expected_.status == NotificationStatus.READ

    def mock_update_notification_status() -> Ok:
        return Ok(expected_)

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    response = test_notification_service.mark_as_unread(authed_context, expected_.id)

    assert response.status == NotificationStatus.UNREAD


def test_service_mark_as_unread_error_on_update__status(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    expected_ = add_mock_notification(
        root_verify_key,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.READ,
    )
    expected_error = "Failed to update  status."

    def mock_update_notificatiion_status(
        credentials: SyftVerifyKey, uid: UID, status: NotificationStatus
    ) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notificatiion_status,
    )

    response = notification_service.mark_as_unread(authed_context, expected_.id)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


# TODO: Fix this test - unsure how to return a LinkedObject Notification.
# Test executes code but does not return a Notification object.
def test_service_resolve_object_success(
    monkeypatch: MonkeyPatch,
    authed_context: AuthedServiceContext,
    linked_object: LinkedObject,
    document_store: DocumentStore,
) -> None:
    test_notification_service = NotificationService(document_store)

    def mock_get_service(linked_obj: LinkedObject) -> NotificationService:
        return test_notification_service

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
        test_notification_service,
        "resolve_link",
        mock_resolve_link,
    )

    response = test_notification_service.resolve_object(authed_context, linked_object)

    assert response is None


def test_service_resolve_object_error_on_resolve_link(
    monkeypatch: MonkeyPatch,
    authed_context: AuthedServiceContext,
    linked_object: LinkedObject,
    document_store: DocumentStore,
) -> None:
    test_notification_service = NotificationService(document_store)
    expected_error = "Failed to resolve link."

    def mock_get_service(linked_obj: LinkedObject) -> NotificationService:
        return test_notification_service

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
        test_notification_service,
        "resolve_link",
        mock_resolve_link,
    )

    response = test_notification_service.resolve_object(authed_context, linked_object)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error


def test_service_clear_success(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_success_ = "All s cleared !!"
    add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )
    inbox_before_delete = test_notification_service.get_all(authed_context)

    assert len(inbox_before_delete) == 1

    def mock_delete_all_for_verify_key(credentials, verify_key) -> Ok:
        return Ok(SyftSuccess.notification)

    monkeypatch.setattr(
        notification_service.stash,
        "delete_all_for_verify_key",
        mock_delete_all_for_verify_key,
    )

    response = test_notification_service.clear(authed_context)
    inbox_after_delete = test_notification_service.get_all(authed_context)

    assert response.notification == expected_success_
    assert len(inbox_after_delete) == 0


def test_service_clear_error_on_delete_all_for_verify_key(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_notification_service = NotificationService(document_store)
    test_stash = NotificationStash(store=document_store)

    expected_error = "Failed to clear s."
    add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )
    inbox_before_delete = test_notification_service.get_all(authed_context)

    assert len(inbox_before_delete) == 1

    def mock_delete_all_for_verify_key(**kwargs) -> Err:
        return Err(expected_error)

    monkeypatch.setattr(
        test_notification_service.stash,
        "delete_all_for_verify_key",
        mock_delete_all_for_verify_key,
    )

    response = test_notification_service.clear(authed_context)
    inbox_after_delete = test_notification_service.get_all(authed_context)

    assert isinstance(response, SyftError)
    assert response.notification == expected_error
    assert len(inbox_after_delete) == 1
