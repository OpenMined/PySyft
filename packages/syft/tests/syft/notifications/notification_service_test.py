# stdlib
from typing import NoReturn

# third party
import pytest
from pytest import MonkeyPatch

# syft absolute
from syft.server.credentials import SyftSigningKey
from syft.server.credentials import SyftVerifyKey
from syft.service.context import AuthedServiceContext
from syft.service.notification.notification_service import NotificationService
from syft.service.notification.notification_stash import NotificationStash
from syft.service.notification.notifications import CreateNotification
from syft.service.notification.notifications import Notification
from syft.service.notification.notifications import NotificationStatus
from syft.service.response import SyftSuccess
from syft.store.document_store import DocumentStore
from syft.store.document_store_errors import StashException
from syft.store.linked_obj import LinkedObject
from syft.types.datetime import DateTime
from syft.types.result import as_result
from syft.types.uid import UID

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


class MockSMTP:
    def __init__(self, smtp_server, smtp_port, timeout):
        self.sent_mail = []
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.timeout = timeout

    def sendmail(self, from_addr, to_addrs, msg):
        self.sent_mail.append((from_addr, to_addrs, msg))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def ehlo(self):
        return True

    def has_extn(self, extn):
        return True

    def login(self, username, password):
        return True

    def starttls(self):
        return True


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
        server_uid=UID(),
        from_user_verify_key=from_user_verify_key,
        to_user_verify_key=to_user_verify_key,
        created_at=DateTime.now(),
        status=status,
    )

    result = notification_stash.set(root_verify_key, mock_notification)
    assert result.is_ok()

    return mock_notification


def test_send_success(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    mock_create_notification: CreateNotification,
    document_store,
) -> None:
    test_notification_service = NotificationService(store=document_store)

    expected_message = mock_create_notification.to(Notification, authed_context)

    @as_result(StashException)
    def mock_set(*args, **kwargs) -> str:
        return expected_message

    monkeypatch.setattr(notification_service.stash, "set", mock_set)
    response = test_notification_service.send(authed_context, mock_create_notification)

    assert isinstance(response, Notification)


def test_send_error_on_set(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    mock_create_notification: CreateNotification,
) -> None:
    test_notification_service = notification_service
    expected_error = "Failed to set notification."

    @as_result(StashException)
    def mock_set(*args, **kwargs) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(notification_service.stash, "set", mock_set)

    with pytest.raises(StashException) as exc:
        test_notification_service.send(authed_context, mock_create_notification)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_get_all_success(
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

    response = test_notification_service.get_all(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    response[0].syft_client_verify_key = None
    response[0].syft_server_location = None
    assert response[0] == expected_message


def test_get_all_error_on_get_all_inbox(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all inbox."

    @as_result(StashException)
    def mock_get_all_inbox_for_verify_key(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_inbox_for_verify_key",
        mock_get_all_inbox_for_verify_key,
    )

    with pytest.raises(StashException) as exc:
        notification_service.get_all(authed_context)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_get_sent_success(
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

    response = test_notification_service.get_all_sent(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    response[0].syft_server_location = None
    response[0].syft_client_verify_key = None
    assert response[0] == expected_message


def test_get_all_error_on_get_all_sent(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all sent."

    @as_result(StashException)
    def mock_get_all_sent_for_verify_key(
        credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_sent_for_verify_key",
        mock_get_all_sent_for_verify_key,
    )

    with pytest.raises(StashException) as exc:
        notification_service.get_all_sent(authed_context)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_get_all_for_status_success(
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

    @as_result(StashException)
    def mock_get_all_by_verify_key_for_status(*args, **kwargs) -> list[Notification]:
        return [expected_message]

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    response = test_notification_service.get_all_for_status(
        authed_context, NotificationStatus.UNREAD
    ).unwrap()

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    assert response[0] == expected_message


def test_error_on_get_all_for_status(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all for status."

    @as_result(StashException)
    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    with pytest.raises(StashException) as exc:
        notification_service.get_all_for_status(
            authed_context,
            NotificationStatus.UNREAD,
        ).unwrap()

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_get_all_read_success(
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

    response = test_notification_service.get_all_read(authed_context)

    assert len(response) == 1
    assert isinstance(response[0], Notification)
    response[0].syft_server_location = None
    response[0].syft_client_verify_key = None
    assert response[0] == expected_message


def test_error_on_get_all_read(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all for status."

    @as_result(StashException)
    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    with pytest.raises(StashException) as exc:
        notification_service.get_all_read(authed_context)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_get_all_unread_success(
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

    response = test_notification_service.get_all_unread(authed_context)
    assert len(response) == 1
    assert isinstance(response[0], Notification)
    response[0].syft_server_location = None
    response[0].syft_client_verify_key = None
    assert response[0] == expected_message


def test_error_on_get_all_unread(
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
) -> None:
    expected_error = "Failed to get all for status."

    @as_result(StashException)
    def mock_get_all_by_verify_key_for_status(
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: NotificationStatus,
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "get_all_by_verify_key_for_status",
        mock_get_all_by_verify_key_for_status,
    )

    with pytest.raises(StashException) as exc:
        notification_service.get_all_unread(authed_context)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_mark_as_read_success(
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

    @as_result(StashException)
    def mock_update_notification_status() -> Notification:
        return expected_message

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    response = test_notification_service.mark_as_read(
        authed_context, expected_message.id
    )

    assert response.status == NotificationStatus.READ


def test_mark_as_read_error_on_update_notification_status(
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
    expected_error = "Failed to update notification status."

    @as_result(StashException)
    def mock_update_notification_status(
        credentials: SyftVerifyKey, uid: UID, status: NotificationStatus
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    with pytest.raises(StashException) as exc:
        notification_service.mark_as_read(authed_context, expected_.id)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_mark_as_unread_success(
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

    expected_notification = add_mock_notification(
        authed_context.credentials,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.READ,
    )

    assert expected_notification.status == NotificationStatus.READ

    as_result(StashException)

    def mock_update_notification_status() -> Notification:
        return expected_notification

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    response = test_notification_service.mark_as_unread(
        authed_context, expected_notification.id
    )

    assert response.status == NotificationStatus.UNREAD


def test_mark_as_unread_error_on_update_notification_status(
    root_verify_key,
    monkeypatch: MonkeyPatch,
    notification_service: NotificationService,
    authed_context: AuthedServiceContext,
    document_store: DocumentStore,
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    expected_notification = add_mock_notification(
        root_verify_key,
        test_stash,
        test_verify_key,
        random_verify_key,
        NotificationStatus.READ,
    )
    expected_error = "Failed to update notification status."

    @as_result(StashException)
    def mock_update_notification_status(
        credentials: SyftVerifyKey, uid: UID, status: NotificationStatus
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        notification_service.stash,
        "update_notification_status",
        mock_update_notification_status,
    )

    with pytest.raises(StashException) as exc:
        notification_service.mark_as_unread(authed_context, expected_notification.id)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


# TODO: Fix this test - unsure how to return a LinkedObject Notification.
# Test executes code but does not return a Notification object.
def test_resolve_object_success(
    monkeypatch: MonkeyPatch,
    authed_context: AuthedServiceContext,
    linked_object: LinkedObject,
    notification_service: NotificationService,
    document_store: DocumentStore,
) -> None:
    test_notification_service = NotificationService(document_store)

    def mock_get_service(linked_obj: LinkedObject) -> NotificationService:
        return test_notification_service

    monkeypatch.setattr(
        authed_context.server,
        "get_service",
        mock_get_service,
    )

    @as_result(StashException)
    def mock_resolve_link(
        context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> None:
        return None

    monkeypatch.setattr(
        test_notification_service,
        "resolve_link",
        mock_resolve_link,
    )

    response = test_notification_service.resolve_object(authed_context, linked_object)

    assert response is None


def test_resolve_object_error_on_resolve_link(
    monkeypatch: MonkeyPatch,
    authed_context: AuthedServiceContext,
    linked_object: LinkedObject,
    document_store: DocumentStore,
    notification_service: NotificationService,
) -> None:
    test_notification_service = NotificationService(document_store)
    expected_error = "Failed to resolve link."

    def mock_get_service(linked_obj: LinkedObject) -> NotificationService:
        return test_notification_service

    monkeypatch.setattr(
        authed_context.server,
        "get_service",
        mock_get_service,
    )

    @as_result(StashException)
    def mock_resolve_link(
        context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        test_notification_service,
        "resolve_link",
        mock_resolve_link,
    )

    with pytest.raises(StashException) as exc:
        test_notification_service.resolve_object(authed_context, linked_object)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error


def test_clear_success(
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
    success_msg = "All notifications cleared!"

    notification = add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )
    inbox_before_delete = test_notification_service.get_all(authed_context)

    assert len(inbox_before_delete) == 1

    @as_result(StashException)
    def mock_delete_all_for_verify_key(credentials, verify_key) -> SyftSuccess:
        return SyftSuccess(message=success_msg, value=[notification.id])

    monkeypatch.setattr(
        notification_service.stash,
        "delete_all_for_verify_key",
        mock_delete_all_for_verify_key,
    )

    response = test_notification_service.clear(authed_context)
    inbox_after_delete = test_notification_service.get_all(authed_context)

    assert response
    assert len(inbox_after_delete) == 0


def test_clear_error_on_delete_all_for_verify_key(
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

    expected_error = "Failed to clear notifications."
    add_mock_notification(
        authed_context.credentials,
        test_stash,
        random_verify_key,
        test_verify_key,
        NotificationStatus.UNREAD,
    )
    inbox_before_delete = test_notification_service.get_all(authed_context)

    assert len(inbox_before_delete) == 1

    @as_result(StashException)
    def mock_delete_all_for_verify_key(**kwargs) -> NoReturn:
        raise StashException(public_message=expected_error)

    monkeypatch.setattr(
        test_notification_service.stash,
        "delete_all_for_verify_key",
        mock_delete_all_for_verify_key,
    )

    with pytest.raises(StashException) as exc:
        test_notification_service.clear(authed_context)

    inbox_after_delete = test_notification_service.get_all(authed_context)

    assert exc.type is StashException
    assert exc.value.public_message == expected_error
    assert len(inbox_after_delete) == 1


# a list of all the mock objects created
mock_smtps = []


def test_send_email(worker, monkeypatch, mock_create_notification, authed_context):
    # stdlib
    import smtplib

    # we use this to have a reference to all the mock objects we create
    def create_smtp(*args, **kwargs):
        # we sum over all the mocks
        global mock_smtps
        res = MockSMTP(*args, **kwargs)
        mock_smtps.append(res)
        return res

    monkeypatch.setattr(smtplib, "SMTP", create_smtp)
    root_client = worker.root_client
    mock_create_notification.to_user_verify_key = root_client.verify_key
    mock_create_notification.from_user_verify_key = root_client.verify_key

    root_client.settings.enable_notifications(
        email_sender="someone@example.com",
        email_port="2525",
        email_server="localhost",
        email_username="someuser",
        email_password="password",
    )

    def emails_sent():
        global mock_smtps
        return sum([len(x.sent_mail) for x in mock_smtps])

    mock_create_notification.to(Notification, authed_context)
    root_client.notifications.send(mock_create_notification)

    assert emails_sent() == 1

    mock_create_notification.id = UID()

    root_client.settings.disable_notifications()
    root_client.notifications.send(mock_create_notification)
    assert emails_sent() == 1

    new_port = "2526"

    root_client.settings.enable_notifications(
        email_sender="someone@example.com",
        email_port=new_port,
        email_server="localhost",
        email_username="someuser",
        email_password="password",
    )

    mock_create_notification.id = UID()
    root_client.notifications.send(mock_create_notification)
    assert emails_sent() == 2
    assert int(mock_smtps[-1].smtp_port) == int(new_port)
