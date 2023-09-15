
from typing import List
from typing import Optional
from typing import Union

from ...serde.serializable import serializable
from ...types.uid import UID
from ...util.telemetry import instrument
from ...store.document_store import DocumentStore
from ..service import AbstractService
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from .log_stash import LogStash
from ..service import service_method
from .log import SyftLog
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL

@instrument
@serializable()
class LogService(AbstractService):
    store: DocumentStore
    stash: LogStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = LogStash(store=store)
        
    @service_method(path="log.add", name="add", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def add(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        new_log = SyftLog(id=uid)
        result = self.stash.set(context.credentials, new_log)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result
        
    @service_method(path="log.append", name="append", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def append(
        self, context: AuthedServiceContext, uid: UID, new_str: str
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))
        new_log = result.ok()
        new_log.append(new_str)
        result = self.stash.update(context.credentials, new_log)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Log Append successful!")
        
    @service_method(path="log.get", name="get", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()
    
    @service_method(path="log.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()
        
    
    @service_method(path="log.delete", name="delete", roles=DATA_SCIENTIST_ROLE_LEVEL)
    def delete(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())