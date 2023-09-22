# third party
import pytest
from pytest import MonkeyPatch
from result import Err

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.service.notification.notification_stash import (
    OrderByCreatedAtTimeStampPartitionKey,
)
from syft.service.notification.notification_stash import FromUserVerifyKeyPartitionKey
from syft.service.notification.notification_stash import NotificationStash
from syft.service.notification.notification_stash import StatusPartitionKey
from syft.service.notification.notification_stash import ToUserVerifyKeyPartitionKey
from syft.service.notification.notifications import Notification
from syft.service.notification.notifications import NotificationExpiryStatus
from syft.service.notification.notifications import NotificationStatus
from syft.types.datetime import DateTime
from syft.types.uid import UID

test_signing_key_string = (
    "2dc2f0fc70a22d082488d5370a5e222437bd2e4eef6dc9f676f96b2218e7a3d5"
)

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


def add_mock_notification(
    root_verify_key,
    notification_stash: NotificationStash,
    from_user_verify_key: SyftVerifyKey,
    to_user_verify_key: SyftVerifyKey,
) -> Notification:
    # prepare: add mock notification

    mock_notification = Notification(
        subject="test_notification",
        node_uid=UID(),
        from_user_verify_key=from_user_verify_key,
        to_user_verify_key=to_user_verify_key,
        created_at=DateTime.now(),
        status=NotificationStatus.UNREAD,
    )

    # print("*******", mock_notification._repr_debug_())

    result = notification_stash.set(root_verify_key, mock_notification)
    assert result.is_ok()

    return mock_notification


def test_fromuserverifykey_partitionkey() -> None:
    random_verify_key = SyftSigningKey.generate().verify_key

    assert FromUserVerifyKeyPartitionKey.type_ == SyftVerifyKey
    assert FromUserVerifyKeyPartitionKey.key == "from_user_verify_key"

    result = FromUserVerifyKeyPartitionKey.with_obj(random_verify_key)

    assert result.type_ == SyftVerifyKey
    assert result.key == "from_user_verify_key"

    assert result.value == random_verify_key

    signing_key = SyftSigningKey.from_string(test_signing_key_string)
    with pytest.raises(AttributeError):
        FromUserVerifyKeyPartitionKey.with_obj(signing_key)


def test_touserverifykey_partitionkey() -> None:
    random_verify_key = SyftSigningKey.generate().verify_key

    assert ToUserVerifyKeyPartitionKey.type_ == SyftVerifyKey
    assert ToUserVerifyKeyPartitionKey.key == "to_user_verify_key"

    result = ToUserVerifyKeyPartitionKey.with_obj(random_verify_key)

    assert result.type_ == SyftVerifyKey
    assert result.key == "to_user_verify_key"
    assert result.value == random_verify_key

    signing_key = SyftSigningKey.from_string(test_signing_key_string)
    with pytest.raises(AttributeError):
        ToUserVerifyKeyPartitionKey.with_obj(signing_key)


def test_status_partitionkey() -> None:
    assert StatusPartitionKey.key == "status"
    assert StatusPartitionKey.type_ == NotificationStatus

    result1 = StatusPartitionKey.with_obj(NotificationStatus.UNREAD)
    result2 = StatusPartitionKey.with_obj(NotificationStatus.READ)

    assert result1.type_ == NotificationStatus
    assert result1.key == "status"
    assert result1.value == NotificationStatus.UNREAD
    assert result2.type_ == NotificationStatus
    assert result2.key == "status"
    assert result2.value == NotificationStatus.READ

    notification_expiry_status_auto = NotificationExpiryStatus(0)

    with pytest.raises(AttributeError):
        StatusPartitionKey.with_obj(notification_expiry_status_auto)


def test_orderbycreatedattimestamp_partitionkey() -> None:
    random_datetime = DateTime.now()

    assert OrderByCreatedAtTimeStampPartitionKey.key == "created_at"
    assert OrderByCreatedAtTimeStampPartitionKey.type_ == DateTime

    result = OrderByCreatedAtTimeStampPartitionKey.with_obj(random_datetime)

    assert result.type_ == DateTime
    assert result.key == "created_at"
    assert result.value == random_datetime


def test_get_all_inbox_for_verify_key(root_verify_key, document_store) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    response = test_stash.get_all_inbox_for_verify_key(
        root_verify_key, random_verify_key
    )

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    # list of mock notifications
    notification_list = []

    for _ in range(5):
        mock_notification = add_mock_notification(
            root_verify_key, test_stash, test_verify_key, random_verify_key
        )
        notification_list.append(mock_notification)

    # returned list of notifications from stash that's sorted by created_at
    response2 = test_stash.get_all_inbox_for_verify_key(
        root_verify_key, random_verify_key
    )

    assert response2.is_ok()

    result = response2.ok()
    assert len(response2.value) == 5

    for notification in notification_list:
        # check if all notifications are present in the result
        assert notification in result

    with pytest.raises(AttributeError):
        test_stash.get_all_inbox_for_verify_key(root_verify_key, random_signing_key)

    # assert that the returned list of notifications is sorted by created_at
    sorted_notification_list = sorted(result, key=lambda x: x.created_at)
    assert result == sorted_notification_list


def test_get_all_sent_for_verify_key(root_verify_key, document_store) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    response = test_stash.get_all_sent_for_verify_key(root_verify_key, test_verify_key)

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    mock_notification = add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    response2 = test_stash.get_all_sent_for_verify_key(root_verify_key, test_verify_key)

    assert response2.is_ok()

    result = response2.ok()
    assert len(response2.value) == 1

    assert result[0] == mock_notification

    with pytest.raises(AttributeError):
        test_stash.get_all_sent_for_verify_key(root_verify_key, random_signing_key)


def test_get_all_for_verify_key(root_verify_key, document_store) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    query_key = FromUserVerifyKeyPartitionKey.with_obj(test_verify_key)
    test_stash = NotificationStash(store=document_store)

    response = test_stash.get_all_for_verify_key(
        root_verify_key, random_verify_key, query_key
    )

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    mock_notification = add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    query_key2 = FromUserVerifyKeyPartitionKey.with_obj(
        mock_notification.from_user_verify_key
    )
    response_from_verify_key = test_stash.get_all_for_verify_key(
        root_verify_key, mock_notification.from_user_verify_key, query_key2
    )
    assert response_from_verify_key.is_ok()

    result = response_from_verify_key.ok()
    assert len(result) == 1

    assert result[0] == mock_notification

    response_from_verify_key_string = test_stash.get_all_for_verify_key(
        root_verify_key, test_verify_key_string, query_key2
    )

    assert response_from_verify_key_string.is_ok()

    result = response_from_verify_key_string.ok()
    assert len(result) == 1


def test_get_all_by_verify_key_for_status(root_verify_key, document_store) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    response = test_stash.get_all_by_verify_key_for_status(
        root_verify_key, random_verify_key, NotificationStatus.READ
    )

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    mock_notification = add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    response2 = test_stash.get_all_by_verify_key_for_status(
        root_verify_key, mock_notification.to_user_verify_key, NotificationStatus.UNREAD
    )
    assert response2.is_ok()

    result = response2.ok()
    assert len(result) == 1

    assert result[0] == mock_notification

    with pytest.raises(AttributeError):
        test_stash.get_all_by_verify_key_for_status(
            root_verify_key, random_signing_key, NotificationStatus.UNREAD
        )


def test_update_notification_status(root_verify_key, document_store) -> None:
    random_uid = UID()
    random_verify_key = SyftSigningKey.generate().verify_key
    test_stash = NotificationStash(store=document_store)
    expected_error = Err(f"No notification exists for id: {random_uid}")

    response = test_stash.update_notification_status(
        root_verify_key, uid=random_uid, status=NotificationStatus.READ
    )

    assert response.is_err()
    assert response == expected_error

    mock_notification = add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    assert mock_notification.status == NotificationStatus.UNREAD

    response2 = test_stash.update_notification_status(
        root_verify_key, uid=mock_notification.id, status=NotificationStatus.READ
    )

    assert response2.is_ok()

    result = response2.ok()
    assert result.status == NotificationStatus.READ

    notification_expiry_status_auto = NotificationExpiryStatus(0)
    with pytest.raises(AttributeError):
        test_stash.pdate_notification_status(
            root_verify_key,
            uid=mock_notification.id,
            status=notification_expiry_status_auto,
        )


def test_update_notification_status_error_on_get_by_uid(
    root_verify_key, monkeypatch: MonkeyPatch, document_store
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    random_uid = UID()
    test_stash = NotificationStash(store=document_store)

    def mock_get_by_uid(root_verify_key, uid: random_uid) -> Err:
        return Err(None)

    monkeypatch.setattr(
        test_stash,
        "get_by_uid",
        mock_get_by_uid,
    )

    add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    response = test_stash.update_notification_status(
        root_verify_key, random_verify_key, NotificationStatus.READ
    )

    assert response is None


def test_delete_all_for_verify_key(root_verify_key, document_store) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    response = test_stash.delete_all_for_verify_key(root_verify_key, test_verify_key)

    assert response.is_ok()

    result = response.ok()
    assert result is True

    add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    inbox_before = test_stash.get_all_inbox_for_verify_key(
        root_verify_key, random_verify_key
    ).value
    assert len(inbox_before) == 1

    response2 = test_stash.delete_all_for_verify_key(root_verify_key, random_verify_key)

    assert response2.is_ok()

    result = response2.ok()
    assert result is True

    inbox_after = test_stash.get_all_inbox_for_verify_key(
        root_verify_key, random_verify_key
    ).value
    assert len(inbox_after) == 0

    with pytest.raises(AttributeError):
        test_stash.delete_all_for_verify_key(root_verify_key, random_signing_key)


def test_delete_all_for_verify_key_error_on_get_all_inbox_for_verify_key(
    root_verify_key, monkeypatch: MonkeyPatch, document_store
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)

    def mock_get_all_inbox_for_verify_key(
        root_verify_key, verify_key: random_verify_key
    ) -> Err:
        return Err(None)

    monkeypatch.setattr(
        test_stash,
        "get_all_inbox_for_verify_key",
        mock_get_all_inbox_for_verify_key,
    )

    response = test_stash.delete_all_for_verify_key(root_verify_key, random_verify_key)

    assert response == Err(None)


def test_delete_all_for_verify_key_error_on_delete_by_uid(
    root_verify_key, monkeypatch: MonkeyPatch, document_store
) -> None:
    random_signing_key = SyftSigningKey.generate()
    random_verify_key = random_signing_key.verify_key
    test_stash = NotificationStash(store=document_store)
    mock_notification = add_mock_notification(
        root_verify_key, test_stash, test_verify_key, random_verify_key
    )

    def mock_delete_by_uid(root_verify_key, uid=mock_notification.id) -> Err:
        return Err(None)

    monkeypatch.setattr(
        test_stash,
        "delete_by_uid",
        mock_delete_by_uid,
    )

    response = test_stash.delete_all_for_verify_key(
        root_verify_key, random_verify_key
    ).value

    assert response is None
