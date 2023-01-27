# future
from __future__ import annotations

# stdlib
from enum import Enum
from typing import Any
from typing import Dict
from typing import Iterable
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
    keys: Union[str, Tuple[str, ...]]
    types: Union[type, Tuple[type, ...]]

    @property
    def pairs(self) -> Iterable[str, type]:
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
    values: Tuple[Any, ...]

    @property
    def pairs(self) -> Iterable[str, type]:
        # make sure we always return Tuple's even if theres a single value
        _keys = (
            self.primary_key.keys
            if isinstance(self.primary_key.keys, (tuple, list))
            else (self.primary_key.keys,)
        )
        _values = (
            self.values if isinstance(self.values, (tuple, list)) else (self.values,)
        )
        return zip(_keys, _values)

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


UIDPrimaryKey = PrimaryKey(keys="id", types=UID)


class CollectionSettings(SyftBaseModel):
    name: str
    primary_key: PrimaryKey


class PrimaryKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


class BaseCollection:
    def __init__(self, settings: CollectionSettings) -> None:
        self.data = {}
        self.primary_key_cols = {}
        self.primary_key_cols["id"] = {}
        for pk_key, _ in settings.primary_key.pairs:
            self.primary_key_cols[pk_key] = {}

    def validate_primary_keys(self, uid: UID, pk: PrimaryKeyObject) -> PrimaryKeyCheck:
        matches = []
        pairs = pk.pairs
        for (pk_key, pk_value) in pairs:
            pk_col = self.primary_key_cols[pk_key]
            if pk_value in pk_col and pk_col[pk_value] == uid:
                matches.append(pk_key)

        if len(matches) == 0:
            return PrimaryKeyCheck.EMPTY
        elif len(matches) == len(pairs):
            return PrimaryKeyCheck.MATCHES
        print("ERROR: matches", matches, pairs)
        return PrimaryKeyCheck.ERROR

    def set(
        self,
        pk: PrimaryKeyObject,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        try:
            uid = obj.id
            pk_check = self.validate_primary_keys(uid=uid, pk=pk)
            if pk_check == PrimaryKeyCheck.EMPTY:
                # write set code
                pass
            if pk_check != PrimaryKeyCheck.ERROR:
                self.data[uid] = obj
        except Exception:
            return Err(f"Failed to write primary_key {pk}")
        return Ok(obj)

    def get(self, pk: PrimaryKeyObject) -> Result[SyftObject, str]:
        try:
            # todo: add id to pk subtype?
            # pk_check = self.validate_primary_keys(uid=uid, pk=pk)
            # if pk_check == PrimaryKeyCheck.MATCHES:
            #     obj = self.data[uid]
            return None
        except Exception:
            return Err(f"Failed to read primary_key {pk}")
        # return Ok(obj)

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def update(self, uid: UID, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def delete(self) -> Result[bool, str]:
        pass


class DocumentStore:
    collections: Dict[str, BaseCollection]
    collection_type: BaseCollection

    def __init__(self) -> None:
        self.collections = {}

    def collection(self, settings: CollectionSettings) -> BaseCollection:
        if settings.name not in self.collections:
            self.collections[settings.name] = self.collection_type(settings=settings)
        return self.collections[settings.name]


class BaseStash:
    object_type: SyftObject
    settings: CollectionSettings
    collection: BaseCollection

    def primary_key(self, obj: SyftObject) -> PrimaryKeyObject:
        return type(self).settings.primary_key.with_obj(obj)

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.collection = store.collection(type(self).settings)

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
