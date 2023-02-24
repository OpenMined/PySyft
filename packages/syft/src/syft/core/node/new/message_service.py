# stdlib
from typing import List
from typing import Union

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .message_stash import MessageStash
from .messages import CreateMessage
from .messages import LinkedObject
from .messages import Message
from .messages import MessageStatus
from .response import SyftError
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
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
        self, context: AuthedServiceContext, message: CreateMessage
    ) -> Union[Message, SyftError]:
        """Send a new message"""

        new_message = message.to(Message, context=context)

        print("New Message:", new_message)
        result = self.stash.set(new_message)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(
        path="messages.get_all",
        name="get_all",
    )
    def get_all(self, context: AuthedServiceContext) -> Union[List[Message], SyftError]:
        print("Context Credentials:", context.credentials)
        result = self.stash.get_all_inbox_for_verify_key(verify_key=context.credentials)
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
            verify_key=context.credentials, status=status
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
            uid=uid, status=MessageStatus.DELIVERED
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


TYPE_TO_SERVICE[Message] = MessageService
SERVICE_TO_TYPES[MessageService].update({Message})
