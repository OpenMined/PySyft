# stdlib
from typing import Any
from typing import List
from typing import Union

# relative
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .context import AuthedServiceContext
from .document_store import DocumentStore
from .response import SyftError
from .response import SyftSuccess
from .service import AbstractService
from .service import service_method
from .user_code import SubmitUserCode
from .user_code import UserCode
from .user_code import execute_byte_code
from .user_code_stash import UserCodeStash


@instrument
@serializable(recursive_serde=True)
class UserCodeService(AbstractService):
    store: DocumentStore
    stash: UserCodeStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = UserCodeStash(store=store)

    @service_method(path="code.add", name="add")
    def add(
        self, context: AuthedServiceContext, code: SubmitUserCode
    ) -> Union[SyftSuccess, SyftError]:
        """Add User Code"""
        result = self.stash.set(code.to(UserCode, context=context))
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="User Code Submitted")

    @service_method(path="code.get_all", name="get_all")
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[UserCode], SyftError]:
        """Get a Dataset"""
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code.get_by_id", name="get_by_id")
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get a User Code Item"""
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code.get_all_for_user", name="get_all_for_user")
    def get_all_for_user(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        """Get All User Code Items for User's VerifyKey"""
        # TODO: replace with incoming user context and key
        result = self.stash.get_all()
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="code.call", name="call")
    def call(
        self, context: AuthedServiceContext, uid: UID, **kwargs: Any
    ) -> Union[SyftSuccess, SyftError]:
        """Call a User Code Function"""
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            code_item = result.ok()
            exec_result = execute_byte_code(code_item, kwargs)
            return exec_result.result
        return SyftError(message=result.err())
