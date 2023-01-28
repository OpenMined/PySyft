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
    key: str
    type_: type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self.key == other.key and self.type_ == other.type_
        return self == other

    def with_obj(self, obj: SyftObject) -> QueryKey:
        return QueryKey.from_obj(primary_key=self, obj=obj)


class PrimaryKeys(BaseModel):
    pks: Union[PrimaryKey, Tuple[PrimaryKey, ...]]

    @property
    def all(self) -> Iterable[PrimaryKey]:
        # make sure we always return Tuple's even if theres a single value
        _keys = self.pks if isinstance(self.pks, (tuple, list)) else (self.pks,)
        return _keys

    def with_obj(self, obj: SyftObject) -> QueryKeys:
        return QueryKeys.from_obj(primary_keys=self, obj=obj)

    def with_tuple(self, *args: Tuple[Any, ...]) -> QueryKeys:
        return QueryKeys.from_tuple(primary_keys=self, args=args)

    def make(self, *obj_arg: Union[SyftObject, Tuple[Any, ...]]) -> QueryKeys:
        if isinstance(obj_arg, SyftObject):
            return self.with_obj(obj_arg)
        else:
            return self.with_tuple(*obj_arg)


class QueryKey(PrimaryKey):
    value: Any

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return (
                self.key == other.key
                and self.type_ == other.type_
                and self.value == other.value
            )
        return self == other

    @staticmethod
    def from_obj(primary_key: PrimaryKey, obj: SyftObject) -> List[Any]:
        pk_key = primary_key.key
        pk_type = primary_key.type_

        if isinstance(obj, pk_type):
            pk_value = obj
        else:
            pk_value = getattr(obj, pk_key)

        if not isinstance(pk_value, pk_type):
            raise Exception(
                f"PrimaryKey {pk_value} of type {type(pk_value)} must be {pk_type}."
            )
        return QueryKey(key=pk_key, type_=pk_type, value=pk_value)


class PrimaryKeysWithUID(PrimaryKeys):
    uid_pk: PrimaryKey

    @property
    def all(self) -> Iterable[PrimaryKey]:
        all_keys = self.pks if isinstance(self.pks, (tuple, list)) else [self.pks]
        if self.uid_pk not in all_keys:
            all_keys.insert(0, self.uid_pk)
        return all_keys


class QueryKeys(SyftBaseModel):
    qks: Union[QueryKey, Tuple[QueryKey, ...]]

    @property
    def all(self) -> Iterable[QueryKey]:
        # make sure we always return Tuple's even if theres a single value
        _keys = self.qks if isinstance(self.qks, (tuple, list)) else (self.qks,)
        return _keys

    @staticmethod
    def from_obj(primary_keys: PrimaryKeys, obj: SyftObject) -> List[Any]:
        qks = []
        for primary_key in primary_keys.all:
            pk_key = primary_key.key
            pk_type = primary_key.type_
            pk_value = getattr(obj, pk_key)
            if not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PrimaryKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
            qk = QueryKey(key=pk_key, type_=pk_type, value=pk_value)
            qks.append(qk)
        return QueryKeys(qks=qks)

    @staticmethod
    def from_tuple(primary_keys: PrimaryKeys, args: Tuple[Any, ...]) -> List[Any]:
        qks = []
        for primary_key, pk_value in zip(primary_keys.all, args):
            pk_key = primary_key.key
            pk_type = primary_key.type_
            if not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PrimaryKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
            qk = QueryKey(key=pk_key, type_=pk_type, value=pk_value)
            print("made qk", qk)
            qks.append(qk)
        return QueryKeys(qks=qks)


UIDPrimaryKey = PrimaryKey(key="id", type_=UID)


class CollectionSettings(SyftBaseModel):
    name: str
    store_key: PrimaryKey = UIDPrimaryKey
    index_keys: PrimaryKeys


class PrimaryKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


class BaseCollection:
    def __init__(self, settings: CollectionSettings) -> None:
        self.data = {}
        self.settings = settings
        self.init_store()

    def init_store(self) -> None:
        self.index_key_cols = {}
        self.index_key_cols[self.settings.store_key.key] = {}
        for primary_key in self.settings.index_keys.all:
            pk_key = primary_key.key
            self.index_key_cols[pk_key] = {}

    def validate_primary_keys(
        self, store_query_key: QueryKey, index_query_keys: QueryKeys
    ) -> PrimaryKeyCheck:
        matches = []
        qks = index_query_keys.all
        for qk in qks:
            pk_key, pk_value = qk.key, qk.value
            if pk_key not in self.index_key_cols:
                raise Exception(
                    f"pk_key: {pk_key} not in index_cols: {self.index_key_cols.keys()}"
                )
            pk_col = self.index_key_cols[pk_key]
            if pk_value in pk_col and pk_col[pk_value] == store_query_key.value:
                matches.append(pk_key)

        if len(matches) == 0:
            return PrimaryKeyCheck.EMPTY
        elif len(matches) == len(qks):
            return PrimaryKeyCheck.MATCHES

        return PrimaryKeyCheck.ERROR

    def set_data_and_keys(
        self, store_query_key: QueryKey, index_query_keys: QueryKeys, obj: SyftObject
    ) -> None:
        # we should lock
        qks = index_query_keys.all
        for qk in qks:
            pk_key, pk_value = qk.key, qk.value
            pk_col = self.index_key_cols[pk_key]
            pk_col[pk_value] = store_query_key.value
            self.index_key_cols[pk_key] = pk_col
        self.index_key_cols[store_query_key.key][
            store_query_key.value
        ] = store_query_key.value
        self.data[store_query_key.value] = obj

    def set(
        self,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        try:
            store_query_key = self.settings.store_key.with_obj(obj)
            exists = store_query_key.value in self.data
            index_query_keys = self.settings.index_keys.with_obj(obj)

            pk_check = self.validate_primary_keys(
                store_query_key=store_query_key, index_query_keys=index_query_keys
            )
            if not exists and pk_check == PrimaryKeyCheck.EMPTY:
                self.set_data_and_keys(
                    store_query_key=store_query_key,
                    index_query_keys=index_query_keys,
                    obj=obj,
                )

            # if pk_check == PrimaryKeyCheck.EMPTY:
            #     # write set code
            #     pass
            # if pk_check != PrimaryKeyCheck.ERROR:
            #     self.data[uid] = obj
        except Exception as e:
            return Err(f"Failed to write obj {obj}. {e}")
        return Ok(obj)

    def get_all_index(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                pk_key, pk_value = qk.key, qk.value
                pk_col = self.index_key_cols[pk_key]
                if pk_value not in pk_col:
                    # must be at least one in all query keys
                    return Ok([])
                store_value = pk_col[pk_value]
                subsets.append({store_value})

            if len(subsets) == 0:
                return Ok([])
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)

            matches = []
            for key in subset:
                matches.append(self.data[key])

            return Ok(matches)
        except Exception:
            return Err(f"Failed to query with {qks}")

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

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.collection = store.collection(type(self).settings)

    def set(self, obj: BaseStash.object_type) -> Result[BaseStash.object_type, str]:
        result = self.collection.set(obj=obj)
        if result.is_ok():
            return result.ok()
        return result.err()

    def get_all_index(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[BaseStash.object_type, str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=qks)
        result = self.collection.get_all_index(qks=qks)

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
