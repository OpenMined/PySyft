# stdlib
from typing import List
from typing import Union

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .credentials import SyftVerifyKey
from .document_store import DocumentStore
from .message_stash import MessageStash
from .messages import Message
from .messages import MessageDelivered
from .messages import MessageStatus
from .response import SyftError
from .response import SyftSuccess
from .service import AbstractService
from .service import service_method


@instrument
@serializable(recursive_serde=True)
class MessageService(AbstractService):
    store: DocumentStore
    stash: MessageStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = MessageStash(store=store)

    @service_method(path="messages.send", name="send")
    def send(
        self, context: AuthedServiceContext, message: Message
    ) -> Union[Message, SyftError]:
        """Send a new message"""

        result = self.stash.set(message)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(
        path="messages.get_all_for_verify_key",
        name="get_all_for_verify_key",
    )
    def get_all_for_verify_key(
        self, context: AuthedServiceContext, verify_key: SyftVerifyKey
    ) -> Union[List[Message], SyftError]:
        result = self.stash.get_all_for_verify_key(verify_key=verify_key)
        if result.err():
            return SyftError(message=str(result.err()))
        messages = result.ok()
        return messages

    @service_method(
        path="messages.get_all_for_status",
        name="get_all_for_status",
    )
    def get_all_for_status(
        self,
        context: AuthedServiceContext,
        verify_key: SyftVerifyKey,
        status: MessageStatus,
    ) -> Union[List[Message], SyftError]:
        result = self.stash.get_all_by_verify_key_for_status(
            verify_key=verify_key, status=status
        )
        if result.err():
            return SyftError(message=str(result.err()))
        messages = result.ok()
        return messages

    @service_method(path="messages.mark_as_delivered", name="mark_as_delivered")
    def mark_as_delivered(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(uid=uid)

        if result.is_err():
            return SyftError(message=str(result.err()))

        message = result.ok()
        message = message.to(MessageDelivered)
        self.stash.update(obj=message)
