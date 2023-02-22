# stdlib
from typing import List
from typing import Union

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .request import Request
from .request import RequestStatus
from .request import SubmitRequest
from .request_stash import RequestStash
from .response import SyftError
from .response import SyftSuccess
from .service import AbstractService
from .service import SERVICE_TO_TYPES
from .service import TYPE_TO_SERVICE
from .service import service_method


@instrument
@serializable(recursive_serde=True)
class RequestService(AbstractService):
    store: DocumentStore
    stash: RequestStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = RequestStash(store=store)

    @service_method(path="request.submit", name="submit")
    def submit(
        self, context: AuthedServiceContext, request: SubmitRequest
    ) -> Union[Request, SyftError]:
        """Submit a Request"""
        result = self.stash.set(request.to(Request, context=context))

        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()

    @service_method(path="request.get_all", name="get_all")
    def get_all(self, context: AuthedServiceContext) -> Union[List[Request], SyftError]:
        result = self.stash.get_all()
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
        request = self.stash.get_by_uid(uid)
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
