# stdlib
import logging

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.linked_obj import LinkedObject
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..notification.email_templates import EmailTemplate
from ..notification.email_templates import RequestEmailTemplate
from ..notification.email_templates import RequestUpdateEmailTemplate
from ..notification.notification_service import CreateNotification
from ..notifier.notifier_enums import NOTIFIERS
from ..notifier.notifier_service import RateLimitException
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import ADMIN_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from .request import Change
from .request import Request
from .request import RequestInfo
from .request import RequestInfoFilter
from .request import RequestStatus
from .request import SubmitRequest
from .request_stash import RequestStash

logger = logging.getLogger(__name__)


@serializable(canonical_name="RequestService", version=1)
class RequestService(AbstractService):
    stash: RequestStash

    def __init__(self, store: DBManager) -> None:
        self.stash = RequestStash(store=store)

    @service_method(path="request.submit", name="submit", roles=GUEST_ROLE_LEVEL)
    def submit(
        self,
        context: AuthedServiceContext,
        request: SubmitRequest,
        send_message: bool = True,
        reason: str | None = "",
    ) -> Request:
        """Submit a Request"""
        request = request.to(Request, context=context)
        request = self.stash.set(
            context.credentials,
            request,
        ).unwrap()

        root_verify_key = context.server.services.user.root_verify_key

        if send_message:
            message_subject = f"Result to request {str(request.id)[:4]}...{str(request.id)[-3:]}\
                has been successfully deposited."
            self._send_email_notification(
                context=context,
                message_subject=message_subject if not reason else reason,
                request=request,
                to_user_verify_key=root_verify_key,
                email_template=RequestEmailTemplate,
            )
        return request

    @service_method(
        path="request.get_by_uid", name="get_by_uid", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_by_uid(self, context: AuthedServiceContext, uid: UID) -> Request:
        return self.stash.get_by_uid(context.credentials, uid).unwrap()

    @service_method(
        path="request.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_all(self, context: AuthedServiceContext) -> list[Request]:
        requests = self.stash.get_all(context.credentials).unwrap()
        # TODO remove once sorting is handled by the stash
        requests.sort(key=lambda x: (x.request_time, x.id), reverse=True)

        return requests

    # DIRTY METHOD: DELETE AFTER DATABASE UPGRADE
    @service_method(
        path="request.get_all_approved",
        name="get_all_approved",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_approved(self, context: AuthedServiceContext) -> list[Request]:
        requests = self.stash.get_all(context.credentials).unwrap()
        # TODO remove once sorting is handled by the stash
        requests = [
            request
            for request in requests
            if request.get_status(context) == RequestStatus.APPROVED
        ]
        requests.sort(key=lambda x: (x.request_time, x.id), reverse=True)

        return requests

    # DIRTY METHOD: DELETE AFTER DATABASE UPGRADE
    @service_method(
        path="request.get_all_rejected",
        name="get_all_rejected",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_rejected(self, context: AuthedServiceContext) -> list[Request]:
        requests = self.stash.get_all(context.credentials).unwrap()
        # TODO remove once sorting is handled by the stash
        requests = [
            request
            for request in requests
            if request.get_status(context) == RequestStatus.REJECTED
        ]
        requests.sort(key=lambda x: (x.request_time, x.id), reverse=True)

        return requests

    # DIRTY METHOD: DELETE AFTER DATABASE UPGRADE
    @service_method(
        path="request.get_all_pending",
        name="get_all_pending",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_all_pending(self, context: AuthedServiceContext) -> list[Request]:
        requests = self.stash.get_all(context.credentials).unwrap()
        # TODO remove once sorting is handled by the stash
        requests = [
            request
            for request in requests
            if request.get_status(context) == RequestStatus.PENDING
        ]
        requests.sort(key=lambda x: (x.request_time, x.id), reverse=True)

        return requests

    @service_method(path="request.get_all_info", name="get_all_info")
    def get_all_info(
        self,
        context: AuthedServiceContext,
        page_index: int | None = 0,
        page_size: int | None = 0,
    ) -> list[list[RequestInfo]] | list[RequestInfo]:
        """Get the information of all requests"""
        result = self.stash.get_all(context.credentials).unwrap()
        requests: list[RequestInfo] = []
        for req in result:
            user = context.server.services.user.get_by_verify_key(
                req.requesting_user_verify_key
            ).unwrap()
            message = context.server.services.notification.filter_by_obj(
                context=context, obj_uid=req.id
            ).unwrap()
            requests.append(RequestInfo(user=user, request=req, notification=message))
        if not page_size:
            return requests

        # If chunk size is defined, then split list into evenly sized chunks
        chunked_requests: list[list[RequestInfo]] = [
            requests[i : i + page_size] for i in range(0, len(requests), page_size)
        ]
        if page_index:
            return chunked_requests[page_index]
        else:
            return chunked_requests

    @service_method(path="request.add_changes", name="add_changes")
    def add_changes(
        self, context: AuthedServiceContext, uid: UID, changes: list[Change]
    ) -> Request:
        request = self.stash.get_by_uid(
            credentials=context.credentials, uid=uid
        ).unwrap()
        request.changes.extend(changes)
        return self.save(context=context, request=request)

    @service_method(path="request.filter_all_info", name="filter_all_info")
    def filter_all_info(
        self,
        context: AuthedServiceContext,
        request_filter: RequestInfoFilter,
        page_index: int | None = 0,
        page_size: int | None = 0,
    ) -> list[RequestInfo]:
        """Filter Request"""
        result = self.get_all_info(context)

        requests = list(
            filter(lambda res: (request_filter.name in res.user.name), result)
        )

        # TODO: Move chunking to a function?
        # If chunk size is defined, then split list into evenly sized chunks
        if page_size:
            requests = [
                requests[i : i + page_size] for i in range(0, len(requests), page_size)
            ]
            if page_index is not None:
                # Return the proper slice using chunk_index
                requests = requests[page_index]

        return requests

    @service_method(path="request.apply", name="apply", unwrap_on_success=False)
    def apply(
        self,
        context: AuthedServiceContext,
        uid: UID,
        **kwargs: dict,
    ) -> SyftSuccess:
        request: Request = self.stash.get_by_uid(context.credentials, uid).unwrap()

        context.extra_kwargs = kwargs
        result = request.apply(context=context).unwrap()
        request_notification = context.server.services.notification.filter_by_obj(
            context=context, obj_uid=uid
        ).unwrap()

        if not request.get_status(context) == RequestStatus.PENDING:
            if request_notification is not None:
                context.server.services.notification.mark_as_read(
                    context=context, uid=request_notification.id
                )

                self._send_email_notification(
                    context=context,
                    message_subject=f"Your request ({str(uid)[:4]}) has been approved. ",
                    request=request,
                    to_user_verify_key=request.requesting_user_verify_key,
                    email_template=RequestUpdateEmailTemplate,
                )
        return result

    @as_result(SyftException, RateLimitException)
    def _send_email_notification(
        self,
        *,
        context: AuthedServiceContext,
        request: Request,
        message_subject: str,
        to_user_verify_key: SyftVerifyKey,
        email_template: type[EmailTemplate],
    ) -> None:
        linked_obj = LinkedObject.with_context(request, context=context)
        notification = CreateNotification(
            subject=message_subject,
            from_user_verify_key=context.credentials,
            to_user_verify_key=to_user_verify_key,
            linked_obj=linked_obj,
            notifier_types=[NOTIFIERS.EMAIL],
            email_template=email_template,
        )
        context.server.services.notification.send(
            context=context, notification=notification
        )

    @service_method(path="request.undo", name="undo", unwrap_on_success=False)
    def undo(self, context: AuthedServiceContext, uid: UID, reason: str) -> SyftSuccess:
        request: Request = self.stash.get_by_uid(
            credentials=context.credentials, uid=uid
        ).unwrap()

        context.extra_kwargs["reason"] = reason
        request.undo(context=context)

        self._send_email_notification(
            context=context,
            message_subject=f"Your request ({str(uid)[:4]}) has been denied. ",
            request=request,
            to_user_verify_key=request.requesting_user_verify_key,
            email_template=RequestUpdateEmailTemplate,
        )

        return SyftSuccess(message=f"Request {uid} successfully denied!")

    def save(self, context: AuthedServiceContext, request: Request) -> Request:
        return self.stash.update(context.credentials, request).unwrap()

    @service_method(
        path="request.delete_by_uid", name="delete_by_uid", unwrap_on_success=False
    )
    def delete_by_uid(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        """Delete the request with the given uid."""
        self.stash.delete_by_uid(context.credentials, uid).unwrap()
        return SyftSuccess(message=f"Request with id {uid} deleted.", value=uid)

    @service_method(
        path="request.set_tags",
        name="set_tags",
        roles=ADMIN_ROLE_LEVEL,
    )
    def set_tags(
        self,
        context: AuthedServiceContext,
        request: Request,
        tags: list[str],
    ) -> Request:
        request = self.stash.get_by_uid(context.credentials, request.id).unwrap()
        request.tags = tags
        return self.save(context, request)

    @service_method(path="request.get_by_usercode_id", name="get_by_usercode_id")
    def get_by_usercode_id(
        self, context: AuthedServiceContext, usercode_id: UID
    ) -> list[Request]:
        return self.stash.get_by_usercode_id(context.credentials, usercode_id).unwrap()


TYPE_TO_SERVICE[Request] = RequestService
SERVICE_TO_TYPES[RequestService].update({Request})
