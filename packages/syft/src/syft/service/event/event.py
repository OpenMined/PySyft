from typing import Any, ClassVar, Dict, List, Type

from syft.service.dataset.dataset import Asset, Dataset
from syft.store.linked_obj import LinkedObject
from ...types.syft_object import SyftObject
from ...types.uid import UID
from datetime import datetime
from pydantic import Field

event_handler_registry = {}

def register_event_handler(event_type):
    def inner(method):
        event_handler_registry[event_type.__name__] = method.__name__
        return method

    return inner

class Event(SyftObject):
    creator_user: UID
    creation_date: datetime = Field(default_factory=lambda: datetime.now())

    def handler(self, node):
        method_name = event_handler_registry[self.__class__.__name__]
        return getattr(node, method_name)
    
class CRUDEvent(Event):
    object_type: ClassVar[Type] = Type
    object_id: UID

    @property
    def merge_updates_repr(self):
        return f"{self.updates} for object {self.object_id} by {self.creator}"

class CreateObjectEvent(CRUDEvent):
    @property
    def updated_properties(self):
        return list(self.object_type.__annotations__.keys())

    @property
    def updates(self):
        return {p: getattr(self, p) for p in self.updated_properties}

    @property
    def update_tuples(self):
        return list(self.updates.items())


class UpdateObjectEvent(CRUDEvent):
    updates: Dict[str, Any]

    @property
    def updated_properties(self):
        return list(self.updates.keys())

    @property
    def update_tuples(self):
        return list(self.updates.items())

class CreateDatasetEvent(CRUDEvent):
    object_type: ClassVar[Type] = Dataset

    def execute(self, node):
        handler = self.handler(node)
        handler(
            object_id=self.real.obj_id,
        )