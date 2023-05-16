# stdlib
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from .message_stash import MessageStash
from .messages import CreateMessage
from .messages import LinkedObject
from .messages import Message
from .messages import MessageStatus


@instrument
@serializable()
class MessageService(AbstractService):
    store: DocumentStore
    stash: MessageStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = MessageStash(store=store)

    @service_method(path="messages.send", name="send")
    def send(
        self, context: AuthedServiceContext, message: CreateMessage
    ) -> Union[Message, SyftError]:
        """Send a new message"""

        new_message = message.to(Message, context=context)
        result = self.stash.set(context.credentials, new_message)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="messages.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> Union[List[Message], SyftError]:
        result = self.stash.get_all_inbox_for_verify_key(
            context.credentials, verify_key=context.credentials
        )
        if result.err():
            return SyftError(message=str(result.err()))
        messages = result.ok()
        return messages

    @service_method(path="messages.get_all_sent", name="outbox")
    def get_all_sent(
        self, context: AuthedServiceContext
    ) -> Union[List[Message], SyftError]:
        result = self.stash.get_all_sent_for_verify_key(
            context.credentials, context.credentials
        )
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
        status: MessageStatus,
    ) -> Union[List[Message], SyftError]:
        result = self.stash.get_all_by_verify_key_for_status(
            context.credentials, verify_key=context.credentials, status=status
        )
        if result.err():
            return SyftError(message=str(result.err()))
        messages = result.ok()
        return messages

    @service_method(path="messages.mark_as_delivered", name="mark_as_delivered")
    def mark_as_delivered(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[Message, SyftError]:
        result = self.stash.update_message_status(
            context.credentials, uid=uid, status=MessageStatus.DELIVERED
        )
        if result.is_err():
            return SyftError(message=str(result.err()))

        return result.ok()

    @service_method(path="messages.resolve_object", name="resolve_object")
    def resolve_object(
        self, context: AuthedServiceContext, linked_obj: LinkedObject
    ) -> Union[Message, SyftError]:
        service = context.node.get_service(linked_obj.service_type)
        result = service.resolve_link(context=context, linked_obj=linked_obj)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="messages.clear", name="clear")
    def clear(self, context: AuthedServiceContext) -> Union[SyftError, SyftSuccess]:
        result = self.stash.delete_all_for_verify_key(
            credentials=context.credentials, verify_key=context.credentials
        )
        if result.is_ok():
            return SyftSuccess(message="All messages cleared !!")
        return SyftError(message=str(result.err()))


TYPE_TO_SERVICE[Message] = MessageService
SERVICE_TO_TYPES[MessageService].update({Message})
