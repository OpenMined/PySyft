# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

# third party
from pydantic import BaseModel
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .base import SyftBaseModel


class PrimaryKey(BaseModel):
    keys: Union[str, Tuple[str]]
    types: Union[type, Tuple[type]]

    @property
    def pairs(self) -> Tuple[Tuple[str], Tuple[type]]:
        # make sure we always return Tuple's even if theres a single value
        _keys = self.keys if isinstance(self.keys, (tuple, list)) else (self.keys,)
        _types = self.types if isinstance(self.types, (tuple, list)) else (self.types,)
        return zip(_keys, _types)

    def with_obj(self, obj: SyftObject) -> PrimaryKeyObject:
        return PrimaryKeyObject.from_object(primary_key=self, obj=obj)

    def with_tuple(self, *args: Tuple[Any, ...]) -> PrimaryKeyObject:
        return PrimaryKeyObject.from_tuple(primary_key=self, args=args)

    def make(self, *obj_arg: Union[SyftObject, Tuple[Any, ...]]) -> PrimaryKeyObject:
        if isinstance(obj_arg, SyftObject):
            return self.with_obj(obj_arg)
        else:
            return self.with_tuple(*obj_arg)


class PrimaryKeyObject(SyftBaseModel):
    primary_key: PrimaryKey
    values: Tuple[Any]

    @staticmethod
    def from_object(primary_key: PrimaryKey, obj: SyftObject) -> List[Any]:
        values = []
        for pk_key, pk_type in primary_key.pairs:
            pk_value = getattr(obj, pk_key)
            if not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PrimaryKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
            values.append(pk_value)
        return PrimaryKeyObject(primary_key=primary_key, values=values)

    @staticmethod
    def from_tuple(primary_key: PrimaryKey, args: Tuple[Any, ...]) -> List[Any]:
        values = []
        for (_, pk_type), pk_value in zip(primary_key.pairs, args):
            if not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PrimaryKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
            values.append(pk_value)
        return PrimaryKeyObject(primary_key=primary_key, values=values)

    def __str__(self) -> str:
        return "_".join(str(self.values))


UIDPrimaryKey = PrimaryKey(keys="id", types=UID)


class BaseCollection:
    def __init__(self) -> None:
        self.data = {}
        # self.permissions = {}

    def set(
        self,
        pk: PrimaryKeyObject,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        try:
            self.data[str(pk)] = obj
        except Exception:
            return Err(f"Failed to write primary_key {pk}")
        return Ok(obj)

    def get(self, pk: PrimaryKeyObject) -> Result[SyftObject, str]:
        try:
            obj = self.data[str(pk)]
        except Exception:
            return Err(f"Failed to read primary_key {pk}")
        return Ok(obj)

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def update(self, uid: UID, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def delete(self) -> Result[bool, str]:
        pass


class CollectionSettings(SyftBaseModel):
    name: str
    primary_key: PrimaryKey


class DocumentStore:
    collections: Dict[str, BaseCollection]
    collection_type: BaseCollection

    def __init__(self) -> None:
        self.collections = {}

    def collection(self, settings: CollectionSettings) -> BaseCollection:
        if settings.name not in self.collections:
            self.collections[settings.name] = self.collection_type()
        return self.collections[settings.name]


class BaseStash:
    object_type: SyftObject
    settings: CollectionSettings = CollectionSettings(
        name="Base", primary_key=UIDPrimaryKey
    )
    collection: BaseCollection

    def primary_key(self, obj: SyftObject) -> PrimaryKeyObject:
        return self.settings.primary_key.with_obj(obj)

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.collection = store.collection(self.settings)

    def set(self, obj: BaseStash.object_type) -> Result[BaseStash.object_type, str]:
        result = self.collection.set(pk=self.primary_key(obj), obj=obj)
        if result.is_ok():
            return result.ok()
        return result.err()

    def get(self, pk: PrimaryKeyObject) -> Result[BaseStash.object_type, str]:
        result = self.collection.get(pk=pk)
        if result.is_ok():
            return result.ok()
        return result.err()

    def delete(self, *args, **kwargs) -> Result[bool, str]:
        return self.collection.delete(*args, **kwargs)


# 🟡 TODO 26: the base collection is already a dict collection but we can change it later
class DictCollection(BaseCollection):
    pass


# the base document store is already a dict but we can change it later
class DictDocumentStore(DocumentStore):
    collection_type = DictCollection
