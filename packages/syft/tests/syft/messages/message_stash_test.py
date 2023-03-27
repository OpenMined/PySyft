# third party
import pytest
from result import Err

# syft absolute
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.datetime import DateTime
from syft.core.node.new.message_stash import FromUserVerifyKeyPartitionKey
from syft.core.node.new.message_stash import MessageStash
from syft.core.node.new.message_stash import StatusPartitionKey
from syft.core.node.new.message_stash import ToUserVerifyKeyPartitionKey
from syft.core.node.new.messages import Message
from syft.core.node.new.messages import MessageExpiryStatus
from syft.core.node.new.messages import MessageStatus
from syft.core.node.new.uid import UID

test_signing_key_string = (
    "2dc2f0fc70a22d082488d5370a5e222437bd2e4eef6dc9f676f96b2218e7a3d5"
)

test_verify_key_string = (
    "e143d6cec3c7701e6ca9c6fb83d09e06fdad947742126831a684a111aba41a8c"
)

test_verify_key = SyftVerifyKey.from_string(test_verify_key_string)


@pytest.fixture(autouse=True)
def message_stash(document_store):
    return MessageStash(store=document_store)


def add_mock_message(
    message_stash: MessageStash,
    from_user_verify_key: SyftVerifyKey,
    to_user_verify_key: SyftVerifyKey,
) -> Message:
    # prepare: add mock message

    # signing_key1 = SyftSigningKey.generate()
    # verify_key1 = signing_key1.verify_key
    # signing_key2 = SyftSigningKey.generate()
    # verify_key2 = signing_key2.verify_key
    message_status_undelivered = MessageStatus(0)

    mock_message = Message(
        subject="test_message",
        node_uid=UID(),
        from_user_verify_key=from_user_verify_key,
        to_user_verify_key=to_user_verify_key,
        created_at=DateTime.now(),
        status=message_status_undelivered,
    )

    # print("*******", mock_message._repr_debug_())

    result = message_stash.partition.set(mock_message)
    assert result.is_ok()

    return mock_message


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
    message_status_undelivered = MessageStatus(0)
    message_status_delivered = MessageStatus(1)

    assert StatusPartitionKey.key == "status"
    assert StatusPartitionKey.type_ == MessageStatus

    result1 = StatusPartitionKey.with_obj(message_status_undelivered)
    result2 = StatusPartitionKey.with_obj(message_status_delivered)

    assert result1.type_ == MessageStatus
    assert result1.key == "status"
    assert result1.value == message_status_undelivered
    assert result2.type_ == MessageStatus
    assert result2.key == "status"
    assert result2.value == message_status_delivered

    message_expiry_status_auto = MessageExpiryStatus(0)

    with pytest.raises(AttributeError):
        StatusPartitionKey.with_obj(message_expiry_status_auto)


def test_messagestash_get_all_inbox_for_verify_key(document_store) -> None:
    random_verify_key = SyftSigningKey.generate().verify_key
    test_stash = MessageStash(store=document_store)

    response = test_stash.get_all_inbox_for_verify_key(random_verify_key)

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    add_mock_message(test_stash, test_verify_key, random_verify_key)

    response2 = test_stash.get_all_inbox_for_verify_key(random_verify_key)

    assert response2.is_ok()

    result = response2.ok()
    assert len(response2.value) == 1


def test_messagestash_get_all_sent_for_verify_key(document_store) -> None:
    random_verify_key = SyftSigningKey.generate().verify_key
    test_stash = MessageStash(store=document_store)

    response = test_stash.get_all_sent_for_verify_key(test_verify_key)

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    add_mock_message(test_stash, test_verify_key, random_verify_key)

    response2 = test_stash.get_all_sent_for_verify_key(test_verify_key)

    assert response2.is_ok()

    result = response2.ok()
    assert len(response2.value) == 1


# TODO: Create QueryKey/s that includes more than just one kind of PartitionKey like
# (FromUserVerifyKeyPartitionKey, ToUserVerifyKeyPartitionKey and StatusPartitionKey)?
def test_messagestash_get_all_for_verify_key(document_store) -> None:
    random_verify_key = SyftSigningKey.generate().verify_key
    query_key = FromUserVerifyKeyPartitionKey.with_obj(test_verify_key)
    test_stash = MessageStash(store=document_store)

    response = test_stash.get_all_for_verify_key(random_verify_key, query_key)

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    mock_message = add_mock_message(test_stash, test_verify_key, random_verify_key)

    query_key2 = FromUserVerifyKeyPartitionKey.with_obj(
        mock_message.from_user_verify_key
    )
    response2 = test_stash.get_all_for_verify_key(
        mock_message.from_user_verify_key, query_key2
    )
    assert response2.is_ok()

    result = response2.ok()
    assert len(result) == 1

    assert result[0] == mock_message


def test_messagestash_get_all_by_verify_key_for_status(document_store) -> None:
    random_verify_key = SyftSigningKey.generate().verify_key
    messeage_status_undelivered = MessageStatus(0)
    messeage_status_delivered = MessageStatus(1)
    test_stash = MessageStash(store=document_store)

    response = test_stash.get_all_by_verify_key_for_status(
        random_verify_key, messeage_status_delivered
    )

    assert response.is_ok()

    result = response.ok()
    assert len(result) == 0

    mock_message = add_mock_message(test_stash, test_verify_key, random_verify_key)

    response2 = test_stash.get_all_by_verify_key_for_status(
        mock_message.from_user_verify_key, messeage_status_undelivered
    )
    assert response2.is_ok()

    result = response2.ok()
    assert len(result) == 1

    assert result[0] == mock_message


# TODO: Peter: don't know how to create test for get_by_uid returns err
# lines 81-83 in message_stash.py
def test_messagestash_update_message_status(document_store) -> None:
    random_uid = UID()
    random_verify_key = SyftSigningKey.generate().verify_key
    messeage_status_undelivered = MessageStatus(0)
    messeage_status_delivered = MessageStatus(1)
    test_stash = MessageStash(store=document_store)
    expected_error = Err(f"No message exists for id: {random_uid}")

    response = test_stash.update_message_status(
        uid=random_uid, status=messeage_status_undelivered
    )

    assert response.is_err()
    assert response == expected_error

    mock_message = add_mock_message(test_stash, test_verify_key, random_verify_key)

    assert mock_message.status == messeage_status_undelivered

    response2 = test_stash.update_message_status(
        uid=mock_message.id, status=messeage_status_delivered
    )

    assert response2.is_ok()

    result = response2.ok()
    assert result.status == messeage_status_delivered


# NOTE: Only works when MessageStash class method: delete_all_for_verify_key,
# line 92 in message_stash.py is changed to:
# messages = self.get_all_inbox_for_verify_key(verify_key=verify_key).value


# TODO: Peter: don't know how to create test for delete_by_uid returns err
# lines 95-95 in message_stash.py
def test_messagestash_delete_all_for_verify_key(document_store) -> None:
    random_verify_key = SyftSigningKey.generate().verify_key
    test_stash = MessageStash(store=document_store)

    response = test_stash.delete_all_for_verify_key(test_verify_key)

    assert response.is_ok()

    result = response.ok()
    assert result is True

    add_mock_message(test_stash, test_verify_key, random_verify_key)

    inbox_before = test_stash.get_all_inbox_for_verify_key(random_verify_key).value
    assert len(inbox_before) == 1

    response2 = test_stash.delete_all_for_verify_key(random_verify_key)

    assert response2.is_ok()

    result = response2.ok()
    assert result is True

    inbox_after = test_stash.get_all_inbox_for_verify_key(random_verify_key).value
    assert len(inbox_after) == 0
