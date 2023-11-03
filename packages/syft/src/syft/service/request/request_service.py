# stdlib
from typing import List
from typing import Optional
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
from ..notification.notification_service import CreateNotification
from ..notification.notification_service import NotificationService
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user import UserView
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..user.user_service import UserService
from .request import Change
from .request import Request
from .request import RequestInfo
from .request import RequestInfoFilter
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

    @service_method(path="request.submit", name="submit", roles=GUEST_ROLE_LEVEL)
    def submit(
        self,
        context: AuthedServiceContext,
        request: SubmitRequest,
        send_message: bool = True,
        reason: Optional[str] = "",
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
                    subject_msg = f"Result to request {str(request.id)[:4]}...{str(request.id)[-3:]}\
                        has been successfully deposited."
                    message = CreateNotification(
                        subject=subject_msg if not reason else reason,
                        from_user_verify_key=context.credentials,
                        to_user_verify_key=root_verify_key,
                        linked_obj=link,
                    )
                    method = context.node.get_service_method(NotificationService.send)
                    result = method(context=context, notification=message)
                    if isinstance(result, Notification):
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

    @service_method(path="request.get_all_info", name="get_all_info")
    def get_all_info(
        self,
        context: AuthedServiceContext,
        page_index: Optional[int] = 0,
        page_size: Optional[int] = 0,
    ) -> Union[List[RequestInfo], SyftError]:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        method = context.node.get_service_method(UserService.get_by_verify_key)
        get_message = context.node.get_service_method(NotificationService.filter_by_obj)

        requests = []
        if result.is_ok():
            for req in result.ok():
                user = method(req.requesting_user_verify_key).to(UserView)
                message = get_message(context=context, obj_uid=req.id)
                requests.append(
                    RequestInfo(user=user, request=req, notification=message)
                )

            # If chunk size is defined, then split list into evenly sized chunks
            if page_size:
                requests = [
                    requests[i : i + page_size]
                    for i in range(0, len(requests), page_size)
                ]
                # Return the proper slice using chunk_index
                requests = requests[page_index]

            return requests

        return SyftError(message=result.err())

    @service_method(path="request.add_changes", name="add_changes")
    def add_changes(
        self, context: AuthedServiceContext, uid: UID, changes: List[Change]
    ) -> Union[Request, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)

        if result.is_err():
            return SyftError(
                message=f"Failed to retrieve request with uid: {uid}. Error: {result.err()}"
            )

        request = result.ok()
        request.changes.extend(changes)
        return self.save(context=context, request=request)

    @service_method(path="request.filter_all_info", name="filter_all_info")
    def filter_all_info(
        self,
        context: AuthedServiceContext,
        request_filter: RequestInfoFilter,
        page_index: Optional[int] = 0,
        page_size: Optional[int] = 0,
    ) -> Union[List[RequestInfo], SyftError]:
        """Get a Dataset"""
        result = self.get_all_info(context)
        requests = list(
            filter(lambda res: (request_filter.name in res.user.name), result)
        )

        # If chunk size is defined, then split list into evenly sized chunks
        if page_size:
            requests = [
                requests[i : i + page_size] for i in range(0, len(requests), page_size)
            ]
            # Return the proper slice using chunk_index
            requests = requests[page_index]

        return requests

    @service_method(
        path="request.apply",
        name="apply",
    )
    def apply(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        request = self.stash.get_by_uid(context.credentials, uid)
        if request.is_ok():
            request = request.ok()
            result = request.apply(context=context)

            filter_by_obj = context.node.get_service_method(
                NotificationService.filter_by_obj
            )
            request_notification = filter_by_obj(context=context, obj_uid=uid)

            link = LinkedObject.with_context(request, context=context)
            if not request.status == RequestStatus.PENDING:
                mark_as_read = context.node.get_service_method(
                    NotificationService.mark_as_read
                )
                mark_as_read(context=context, uid=request_notification.id)

                notification = CreateNotification(
                    subject=f"{request.changes} for Request id: {uid} has status updated to {request.status}",
                    to_user_verify_key=request.requesting_user_verify_key,
                    linked_obj=link,
                )
                send_notification = context.node.get_service_method(
                    NotificationService.send
                )
                send_notification(context=context, notification=notification)

            # TODO: check whereever we're return SyftError encapsulate it in Result.
            if hasattr(result, "value"):
                return result.value
            return result
        return request.value

    @service_method(path="request.undo", name="undo")
    def undo(
        self, context: AuthedServiceContext, uid: UID, reason: str
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(credentials=context.credentials, uid=uid)
        if result.is_err():
            return SyftError(
                message=f"Failed to update request: {uid} with error: {result.err()}"
            )

        request = result.ok()
        if request is None:
            return SyftError(message=f"Request with uid: {uid} does not exists.")

        context.extra_kwargs["reason"] = reason
        result = request.undo(context=context)

        if result.is_err():
            return SyftError(
                message=f"Failed to undo Request: <{uid}> with error: {result.err()}"
            )

        link = LinkedObject.with_context(request, context=context)
        message_subject = (
            f"Your request for uid: {uid} has been denied. "
            f"Reason specified by Data Owner: {reason}."
        )

        notification = CreateNotification(
            subject=message_subject,
            to_user_verify_key=request.requesting_user_verify_key,
            linked_obj=link,
        )
        send_notification = context.node.get_service_method(NotificationService.send)
        send_notification(context=context, notification=notification)

        return SyftSuccess(message=f"Request {uid} successfully denied !")

    def save(
        self, context: AuthedServiceContext, request: Request
    ) -> Union[Request, SyftError]:
        result = self.stash.update(context.credentials, request)
        if result.is_ok():
            return result.ok()
        return SyftError(
            message=f"Failed to update Request: <{request.id}>. Error: {result.err()}"
        )


TYPE_TO_SERVICE[Request] = RequestService
SERVICE_TO_TYPES[RequestService].update({Request})
