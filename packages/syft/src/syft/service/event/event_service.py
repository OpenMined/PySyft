from syft.serde.serializable import serializable
from syft.service.context import AuthedServiceContext
from syft.service.event.event_stash import EventStash
from syft.service.response import SyftError, SyftSuccess
from syft.service.service import AbstractService, service_method
from syft.service.user.user_roles import DATA_OWNER_ROLE_LEVEL
from syft.store.document_store import DocumentStore
from syft.types.uid import UID
from syft.util.trace_decorator import instrument
from .event import Event

@instrument
@serializable()
class EventService(AbstractService):
    store: DocumentStore
    stash: EventStash
    
    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = EventStash(store=store)
        
    @service_method(
        path="event.add",
        name="add",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def add(
        self, context: AuthedServiceContext, event: Event, 
    ):
        result = self.stash.set(context.credentials, event)
        if result.is_err():
            return SyftError(message=str(result.err()))
    
        return SyftSuccess(message=f'Great Success!')
    
    
    @service_method(
        path="event.get_by_uid",
        name="get_by_uid",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_by_uid(
        self, context: AuthedServiceContext, uid: UID,    
    ):
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return result.ok()
        
    
    @service_method(
        path="event.get_all",
        name="get_all",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_all(
        self, context: AuthedServiceContext
    ):
        result = self.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=str(result.err()))
    
        return result.ok()
    
    