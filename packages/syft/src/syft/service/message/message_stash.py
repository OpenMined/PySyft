# stdlib
from typing import List

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.uid import UID
from ...util.telemetry import instrument
from .messages import Message
from .messages import MessageStatus

FromUserVerifyKeyPartitionKey = PartitionKey(
    key="from_user_verify_key", type_=SyftVerifyKey
)
ToUserVerifyKeyPartitionKey = PartitionKey(
    key="to_user_verify_key", type_=SyftVerifyKey
)
StatusPartitionKey = PartitionKey(key="status", type_=MessageStatus)


@instrument
@serializable()
class MessageStash(BaseUIDStoreStash):
    object_type = Message
    settings: PartitionSettings = PartitionSettings(
        name=Message.__canonical_name__,
        object_type=Message,
    )

    def get_all_inbox_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[List[Message], str]:
        qks = QueryKeys(
            qks=[
                ToUserVerifyKeyPartitionKey.with_obj(verify_key),
            ]
        )
        return self.get_all_for_verify_key(
            credentials=credentials, verify_key=verify_key, qks=qks
        )

    def get_all_sent_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[List[Message], str]:
        qks = QueryKeys(
            qks=[
                FromUserVerifyKeyPartitionKey.with_obj(verify_key),
            ]
        )
        return self.get_all_for_verify_key(credentials, verify_key=verify_key, qks=qks)

    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey, qks: QueryKeys
    ) -> Result[List[Message], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        return self.query_all(credentials, qks=qks)

    def get_all_by_verify_key_for_status(
        self,
        credentials: SyftVerifyKey,
        verify_key: SyftVerifyKey,
        status: MessageStatus,
    ) -> Result[List[Message], str]:
        qks = QueryKeys(
            qks=[
                ToUserVerifyKeyPartitionKey.with_obj(verify_key),
                StatusPartitionKey.with_obj(status),
            ]
        )
        return self.query_all(credentials, qks=qks)

    def update_message_status(
        self, credentials: SyftVerifyKey, uid: UID, status: MessageStatus
    ) -> Result[Message, str]:
        result = self.get_by_uid(credentials, uid=uid)
        if result.is_err():
            return result.err()

        message = result.ok()
        if message is None:
            return Err(f"No message exists for id: {uid}")
        message.status = status
        return self.update(credentials, obj=message)

    def delete_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[bool, str]:
        result = self.get_all_inbox_for_verify_key(credentials, verify_key=verify_key)
        # If result is an error then return the error
        if result.is_err():
            return result

        # get the list of messages
        messages = result.ok()

        for message in messages:
            result = self.delete_by_uid(credentials, uid=message.id)
            if result.is_err():
                return result
        return Ok(True)
