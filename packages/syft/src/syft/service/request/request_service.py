# stdlib
from typing import List
from typing import Union

# third party
from result import Err
from result import Ok

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.linked_obj import LinkedObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..message.message_service import CreateMessage
from ..message.message_service import Message
from ..message.message_service import MessageService
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_service import UserService
from .request import Request
from .request import RequestStatus
from .request import SubmitRequest
from .request_stash import RequestStash


@instrument
@serializable()
class RequestService(AbstractService):
    store: DocumentStore
    stash: RequestStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = RequestStash(store=store)

    @service_method(path="request.submit", name="submit")
    def submit(
        self,
        context: AuthedServiceContext,
        request: SubmitRequest,
        send_message: bool = True,
    ) -> Union[Request, SyftError]:
        """Submit a Request"""
        try:
            req = request.to(Request, context=context)
            result = self.stash.set(
                context.credentials,
                req,
                add_permissions=[
                    ActionObjectPermission(
                        uid=req.id, permission=ActionPermission.ALL_READ
                    ),
                ],
            )
            if result.is_ok():
                request = result.ok()
                link = LinkedObject.with_context(request, context=context)
                admin_verify_key = context.node.get_service_method(
                    UserService.admin_verify_key
                )

                root_verify_key = admin_verify_key()
                if send_message:
                    message = CreateMessage(
                        subject="Approval Request",
                        from_user_verify_key=context.credentials,
                        to_user_verify_key=root_verify_key,
                        linked_obj=link,
                    )
                    method = context.node.get_service_method(MessageService.send)
                    result = method(context=context, message=message)
                    if isinstance(result, Message):
                        return Ok(request)
                    else:
                        return Err(result)

                return Ok(request)

            if result.is_err():
                return SyftError(message=str(result.err()))
            return result.ok()
        except Exception as e:
            print("Failed to submit Request", e)
            raise e

    @service_method(path="request.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> Union[List[Request], SyftError]:
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        requests = result.ok()
        return requests

    @service_method(path="request.get_all_for_status", name="get_all_for_status")
    def get_all_for_status(
        self, context: AuthedServiceContext, status: RequestStatus
    ) -> Union[List[Request], SyftError]:
        result = self.stash.get_all_for_status(status=status)
        if result.is_err():
            return SyftError(message=str(result.err()))
        requests = result.ok()
        return requests

    @service_method(path="request.apply", name="apply")
    def apply(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        request = self.stash.get_by_uid(context.credentials, uid)
        if request.is_ok():
            request = request.ok()
            result = request.apply(context=context)
            return result.value
        return request.value

    @service_method(path="request.revert", name="revert")
    def revert(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        request = self.stash.get_by_uid(uid)
        if request.is_ok():
            result = request.ok().revert(context=context)
            return result.value
        return request.value


TYPE_TO_SERVICE[Request] = RequestService
SERVICE_TO_TYPES[RequestService].update({Request})
