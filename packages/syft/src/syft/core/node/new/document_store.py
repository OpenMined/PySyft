# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

# third party
from pydantic import BaseModel
from result import Err
from result import Ok
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .base import SyftBaseModel


def first_or_none(result: Any) -> Optional[Any]:
    if hasattr(result, "__len__") and len(result) > 0:
        return Ok(result[0])
    return Ok(None)


@serializable(recursive_serde=True)
class CollectionKey(BaseModel):
    key: str
    type_: type

    def __eq__(self, other: Any) -> bool:
        if type(other) == type(self):
            return self.key == other.key and self.type_ == other.type_
        return False

    def with_obj(self, obj: SyftObject) -> QueryKey:
        return QueryKey.from_obj(collection_key=self, obj=obj)


@serializable(recursive_serde=True)
class CollectionKeys(BaseModel):
    cks: Union[CollectionKey, Tuple[CollectionKey, ...]]

    @property
    def all(self) -> Iterable[CollectionKey]:
        # make sure we always return Tuple's even if theres a single value
        _keys = self.cks if isinstance(self.cks, (tuple, list)) else (self.cks,)
        return _keys

    def with_obj(self, obj: SyftObject) -> QueryKeys:
        return QueryKeys.from_obj(collection_keys=self, obj=obj)

    def with_tuple(self, *args: Tuple[Any, ...]) -> QueryKeys:
        return QueryKeys.from_tuple(collection_keys=self, args=args)

    def add(self, ck: CollectionKey) -> CollectionKeys:
        return CollectionKeys(cks=list(self.all) + [ck])

    @staticmethod
    def from_dict(cks_dict: Dict[str, type]) -> CollectionKeys:
        cks = []
        for k, t in cks_dict.items():
            cks.append(CollectionKey(key=k, type_=t))
        return CollectionKeys(cks=cks)

    def make(self, *obj_arg: Union[SyftObject, Tuple[Any, ...]]) -> QueryKeys:
        if isinstance(obj_arg, SyftObject):
            return self.with_obj(obj_arg)
        else:
            return self.with_tuple(*obj_arg)


@serializable(recursive_serde=True)
class QueryKey(CollectionKey):
    value: Any

    def __eq__(self, other: Any) -> bool:
        if type(other) == type(self):
            return (
                self.key == other.key
                and self.type_ == other.type_
                and self.value == other.value
            )
        return False

    @property
    def collection_key(self) -> CollectionKey:
        return CollectionKey(key=self.key, type_=self.type_)

    @staticmethod
    def from_obj(collection_key: CollectionKey, obj: SyftObject) -> List[Any]:
        ck_key = collection_key.key
        ck_type = collection_key.type_

        if isinstance(obj, ck_type):
            ck_value = obj
        else:
            ck_value = getattr(obj, ck_key)

        if ck_value and not isinstance(ck_value, ck_type):
            raise Exception(
                f"CollectionKey {ck_value} of type {type(ck_value)} must be {ck_type}."
            )
        return QueryKey(key=ck_key, type_=ck_type, value=ck_value)


@serializable(recursive_serde=True)
class CollectionKeysWithUID(CollectionKeys):
    uid_pk: CollectionKey

    @property
    def all(self) -> Iterable[CollectionKey]:
        all_keys = self.cks if isinstance(self.cks, (tuple, list)) else [self.cks]
        if self.uid_pk not in all_keys:
            all_keys.insert(0, self.uid_pk)
        return all_keys


@serializable(recursive_serde=True)
class QueryKeys(SyftBaseModel):
    qks: Union[QueryKey, Tuple[QueryKey, ...]]

    @property
    def all(self) -> Iterable[QueryKey]:
        # make sure we always return Tuple's even if theres a single value
        _keys = self.qks if isinstance(self.qks, (tuple, list)) else (self.qks,)
        return _keys

    @staticmethod
    def from_obj(collection_keys: CollectionKeys, obj: SyftObject) -> QueryKeys:
        qks = []
        for collection_key in collection_keys.all:
            ck_key = collection_key.key
            ck_type = collection_key.type_
            ck_value = getattr(obj, ck_key)
            if ck_value and not isinstance(ck_value, ck_type):
                raise Exception(
                    f"CollectionKey {ck_value} of type {type(ck_value)} must be {ck_type}."
                )
            qk = QueryKey(key=ck_key, type_=ck_type, value=ck_value)
            qks.append(qk)
        return QueryKeys(qks=qks)

    @staticmethod
    def from_tuple(collection_keys: CollectionKeys, args: Tuple[Any, ...]) -> QueryKeys:
        qks = []
        for collection_key, ck_value in zip(collection_keys.all, args):
            ck_key = collection_key.key
            ck_type = collection_key.type_
            if not isinstance(ck_value, ck_type):
                raise Exception(
                    f"CollectionKey {ck_value} of type {type(ck_value)} must be {ck_type}."
                )
            qk = QueryKey(key=ck_key, type_=ck_type, value=ck_value)
            qks.append(qk)
        return QueryKeys(qks=qks)

    @staticmethod
    def from_dict(qks_dict: Dict[str, Any]) -> QueryKeys:
        qks = []
        for k, v in qks_dict.items():
            qks.append(QueryKey(key=k, type_=type(v), value=v))
        return QueryKeys(qks=qks)


UIDCollectionKey = CollectionKey(key="id", type_=UID)


@serializable(recursive_serde=True)
class CollectionSettings(SyftBaseModel):
    name: str
    object_type: type
    store_key: CollectionKey = UIDCollectionKey

    @property
    def unique_keys(self) -> CollectionKeys:
        unique_keys = CollectionKeys.from_dict(
            self.object_type._syft_unique_keys_dict()
        )
        return unique_keys.add(self.store_key)

    @property
    def searchable_keys(self) -> CollectionKeys:
        return CollectionKeys.from_dict(self.object_type._syft_searchable_keys_dict())


@serializable(recursive_serde=True)
class UniqueKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


@instrument
@serializable(recursive_serde=True)
class BaseCollection:
    def __init__(self, settings: CollectionSettings) -> None:
        self.data = {}
        self.settings = settings
        self.init_store()

    def store_query_key(self, obj: Any) -> QueryKey:
        return self.settings.store_key.with_obj(obj)

    def store_query_keys(self, objs: Any) -> QueryKeys:
        return QueryKeys(qks=[self.store_query_key(obj) for obj in objs])

    def init_store(self) -> None:
        self.unique_cks = self.settings.unique_keys.all
        self.unique_keys = {}

        for collection_key in self.unique_cks:
            ck_key = collection_key.key
            self.unique_keys[ck_key] = {}

        self.searchable_cks = self.settings.searchable_keys.all
        self.searchable_keys = {}
        for collection_key in self.searchable_cks:
            ck_key = collection_key.key
            self.searchable_keys[ck_key] = defaultdict(list)

    def validate_collection_keys(
        self, store_query_key: QueryKey, unique_query_keys: QueryKeys
    ) -> UniqueKeyCheck:
        matches = []
        qks = unique_query_keys.all
        for qk in qks:
            ck_key, ck_value = qk.key, qk.value
            if ck_key not in self.unique_keys:
                raise Exception(
                    f"ck_key: {ck_key} not in unique_keys: {self.unique_keys.keys()}"
                )
            ck_col = self.unique_keys[ck_key]
            if ck_value in ck_col and ck_col[ck_value] == store_query_key.value:
                matches.append(ck_key)

        if len(matches) == 0:
            return UniqueKeyCheck.EMPTY
        elif len(matches) == len(qks):
            return UniqueKeyCheck.MATCHES

        return UniqueKeyCheck.ERROR

    def set_data_and_keys(
        self,
        store_query_key: QueryKey,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
        obj: SyftObject,
    ) -> None:
        # we should lock
        uqks = unique_query_keys.all
        for qk in uqks:
            ck_key, ck_value = qk.key, qk.value
            ck_col = self.unique_keys[ck_key]
            ck_col[ck_value] = store_query_key.value
            self.unique_keys[ck_key] = ck_col

        self.unique_keys[store_query_key.key][
            store_query_key.value
        ] = store_query_key.value

        sqks = searchable_query_keys.all
        for qk in sqks:
            ck_key, ck_value = qk.key, qk.value
            ck_col = self.searchable_keys[ck_key]
            ck_col[ck_value].append(store_query_key.value)
            self.searchable_keys[ck_key] = ck_col

        self.data[store_query_key.value] = obj

    def set(
        self,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        try:
            store_query_key = self.settings.store_key.with_obj(obj)
            exists = store_query_key.value in self.data
            unique_query_keys = self.settings.unique_keys.with_obj(obj)
            searchable_query_keys = self.settings.searchable_keys.with_obj(obj)

            ck_check = self.validate_collection_keys(
                store_query_key=store_query_key, unique_query_keys=unique_query_keys
            )
            if not exists and ck_check == UniqueKeyCheck.EMPTY:
                self.set_data_and_keys(
                    store_query_key=store_query_key,
                    unique_query_keys=unique_query_keys,
                    searchable_query_keys=searchable_query_keys,
                    obj=obj,
                )

            # if ck_check == UniqueKeyCheck.EMPTY:
            #     # write set code
            #     pass
            # if ck_check != UniqueKeyCheck.ERROR:
            #     self.data[uid] = obj
        except Exception as e:
            return Err(f"Failed to write obj {obj}. {e}")
        return Ok(obj)

    def remove_keys(
        self,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
    ) -> None:

        uqks = unique_query_keys.all
        for qk in uqks:
            ck_key, ck_value = qk.key, qk.value
            ck_col = self.unique_keys[ck_key]
            ck_col.pop(ck_value, None)

        sqks = searchable_query_keys.all
        for qk in sqks:
            ck_key, ck_value = qk.key, qk.value
            ck_col = self.searchable_keys[ck_key]
            ck_col.pop(ck_value, None)

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        try:
            _original_obj = self.data[qk.value]
            _original_unique_keys = self.settings.unique_keys.with_obj(_original_obj)
            _original_searchable_keys = self.settings.searchable_keys.with_obj(
                _original_obj
            )

            # 🟡 TODO 28: Add locking in this transaction

            # remove old keys
            self.remove_keys(
                unique_query_keys=_original_unique_keys,
                searchable_query_keys=_original_searchable_keys,
            )

            # update the object with new data
            for key, value in dict(obj).items():
                setattr(_original_obj, key, value)

            # update data and keys
            self.set_data_and_keys(
                store_query_key=qk,
                unique_query_keys=self.settings.unique_keys.with_obj(_original_obj),
                searchable_query_keys=self.settings.searchable_keys.with_obj(
                    _original_obj
                ),
                obj=_original_obj,
            )

            return Ok(_original_obj)
        except Exception as e:
            return Err(f"Failed to update obj {obj} with error: {e}")

    def get_all_from_store(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        matches = []
        for qk in qks.all:
            if qk.value in self.data:
                matches.append(self.data[qk.value])
        return Ok(matches)

    def delete_unique_keys_for(self, obj: SyftObject) -> Result[bool, str]:
        for _unique_ck in self.unique_cks:
            qk = _unique_ck.with_obj(obj)
            self.unique_keys[qk.key].pop(qk.value, None)
        return Ok(True)

    def delete_search_keys_for(self, obj: SyftObject) -> Result[bool, str]:
        for _search_ck in self.searchable_cks:
            qk = _search_ck.with_obj(obj)
            self.searchable_keys[qk.key].pop(qk.value, None)
        return Ok(True)

    def get_keys_index(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                ck_key, ck_value = qk.key, qk.value
                if ck_key not in self.unique_keys:
                    return Err(f"Failed to query index with {qk}")
                ck_col = self.unique_keys[ck_key]
                if ck_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_value = ck_col[ck_value]
                subsets.append({store_value})

            if len(subsets) == 0:
                return Ok(set())
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)

            return Ok(subset)
        except Exception as e:
            return Err(f"Failed to query with {qks}. {e}")

    def find_keys_search(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                ck_key, ck_value = qk.key, qk.value
                if ck_key not in self.searchable_keys:
                    return Err(f"Failed to search with {qk}")
                ck_col = self.searchable_keys[ck_key]
                if ck_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_values = ck_col[ck_value]
                subsets.append(set(store_values))

            if len(subsets) == 0:
                return Ok(set())
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)

            return Ok(subset)
        except Exception as e:
            return Err(f"Failed to query with {qks}. {e}")

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def delete(self, qk: QueryKey) -> Result[bool, str]:

        try:
            _obj = self.data.pop(qk.value)
            self.delete_unique_keys_for(_obj)
            self.delete_search_keys_for(_obj)
            return Ok(True)
        except Exception as e:
            return Err(f"Failed to delete with query key {qk} with error: {e}")


@instrument
@serializable(recursive_serde=True)
class DocumentStore:
    collections: Dict[str, BaseCollection]
    collection_type: BaseCollection

    def __init__(self) -> None:
        self.collections = {}

    def collection(self, settings: CollectionSettings) -> BaseCollection:
        if settings.name not in self.collections:
            self.collections[settings.name] = self.collection_type(settings=settings)
        return self.collections[settings.name]


@instrument
class BaseStash:
    object_type: Type[SyftObject]
    settings: CollectionSettings
    collection: BaseCollection

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.collection = store.collection(type(self).settings)

    def set(self, obj: BaseStash.object_type) -> Result[BaseStash.object_type, str]:
        return self.collection.set(obj=obj)

    def get_all(self) -> List[BaseStash.object_type]:
        return list(self.collection.data.values())

    def __len__(self) -> int:
        return len(self.get_all())

    def clear(self) -> None:
        self.collection.data = {}

    def get_all_index(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[BaseStash.object_type, str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=qks)
        result = self.collection.get_keys_index(qks=qks)
        if result.is_ok():
            qks = self.collection.store_query_keys(result.ok())
            objects = self.collection.get_all_from_store(qks=qks)
            return objects
        return Err(result.err())

    def find_all_search(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[BaseStash.object_type, str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=qks)
        result = self.collection.find_keys_search(qks=qks)

        if result.is_ok():
            qks = self.collection.store_query_keys(result.ok())
            objects = self.collection.get_all_from_store(qks=qks)
            return objects
        return Err(result.err())

    def query_all(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[List[BaseStash.object_type], str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=qks)

        unique_keys = []
        searchable_keys = []

        for qk in qks.all:
            ck = qk.collection_key
            if ck in self.collection.unique_cks:
                unique_keys.append(qk)
            elif ck in self.collection.searchable_cks:
                searchable_keys.append(qk)
            else:
                raise Exception(
                    f"{qk} not in {type(self.collection)} unique or searchable keys"
                )

        ids: Optional[Set] = None
        errors = []
        if len(unique_keys) > 0:
            index_results = self.collection.get_keys_index(
                qks=QueryKeys(qks=unique_keys)
            )
            if index_results.is_ok():
                if ids is None:
                    ids = index_results.ok()
                ids = ids.intersection(index_results.ok())
            else:
                errors.append(index_results.err())

        search_results = None
        if len(searchable_keys) > 0:
            search_results = self.collection.find_keys_search(
                qks=QueryKeys(qks=searchable_keys)
            )

            if search_results.is_ok():
                if ids is None:
                    ids = search_results.ok()
                ids = ids.intersection(search_results.ok())
            else:
                errors.append(search_results.err())

        if len(errors) > 0:
            return Err(" ".join(errors))

        qks = self.collection.store_query_keys(ids)
        objects = self.collection.get_all_from_store(qks=qks)
        return objects

    def query_all_kwargs(
        self, **kwargs: Dict[str, Any]
    ) -> Result[List[BaseStash.object_type], str]:
        qks = QueryKeys.from_dict(kwargs)
        return self.query_all(qks=qks)

    def query_one(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[Optional[BaseStash.object_type], str]:
        return self.query_all(qks=qks).and_then(first_or_none)

    def query_one_kwargs(
        self,
        **kwargs: Dict[str, Any],
    ) -> Result[Optional[BaseStash.object_type], str]:
        return self.query_all_kwargs(**kwargs).and_then(first_or_none)

    def find_all(
        self, **kwargs: Dict[str, Any]
    ) -> Result[List[BaseStash.object_type], str]:
        return self.query_all_kwargs(**kwargs)

    def find_one(
        self, **kwargs: Dict[str, Any]
    ) -> Result[Optional[BaseStash.object_type], str]:
        return self.query_one_kwargs(**kwargs)

    def find_and_delete(self, **kwargs: Dict[str, Any]) -> Result[bool, str]:
        obj = self.query_one_kwargs(**kwargs)
        if obj.is_err():
            return obj.err()
        else:
            obj = obj.ok()

        if not obj:
            return Err(f"Object does not exists with kwargs: {kwargs}")
        qk = self.collection.store_query_key(obj)
        return self.delete(qk=qk)

    def delete(self, qk: QueryKey) -> Result[bool, str]:
        return self.collection.delete(qk=qk)

    def update(
        self, obj: BaseStash.object_type
    ) -> Optional[Result[BaseStash.object_type, str]]:

        qks = self.collection.settings.unique_keys.with_obj(obj)
        result = self.collection.get_keys_index(qks=qks)

        if result.is_ok():
            result = result.ok()
            if len(result) < 1:
                return Err(f"No obj found for query keys: {qks}")
            elif len(result) > 1:
                return Err(f"Multiple objects found for query keys: {qks}")

            qk = self.collection.store_query_key(result.pop())
            updated_obj = self.collection.update(qk=qk, obj=obj)
            return updated_obj
        return Err(result.err())


# 🟡 TODO 26: the base collection is already a dict collection but we can change it later
@serializable(recursive_serde=True)
class DictCollection(BaseCollection):
    pass


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class DictDocumentStore(DocumentStore):
    collection_type = DictCollection
