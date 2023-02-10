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
from .response import SyftSuccess


def first_or_none(result: Any) -> Optional[Any]:
    if hasattr(result, "__len__") and len(result) > 0:
        return Ok(result[0])
    return Ok(None)


@serializable(recursive_serde=True)
class PartitionKey(BaseModel):
    key: str
    type_: type

    def __eq__(self, other: Any) -> bool:
        if type(other) == type(self):
            return self.key == other.key and self.type_ == other.type_
        return False

    def with_obj(self, obj: SyftObject) -> QueryKey:
        return QueryKey.from_obj(partition_key=self, obj=obj)


@serializable(recursive_serde=True)
class PartitionKeys(BaseModel):
    pks: Union[PartitionKey, Tuple[PartitionKey, ...]]

    @property
    def all(self) -> Iterable[PartitionKey]:
        # make sure we always return Tuple's even if theres a single value
        _keys = self.pks if isinstance(self.pks, (tuple, list)) else (self.pks,)
        return _keys

    def with_obj(self, obj: SyftObject) -> QueryKeys:
        return QueryKeys.from_obj(partition_keys=self, obj=obj)

    def with_tuple(self, *args: Tuple[Any, ...]) -> QueryKeys:
        return QueryKeys.from_tuple(partition_keys=self, args=args)

    def add(self, pk: PartitionKey) -> PartitionKeys:
        return PartitionKeys(pks=list(self.all) + [pk])

    @staticmethod
    def from_dict(cks_dict: Dict[str, type]) -> PartitionKeys:
        pks = []
        for k, t in cks_dict.items():
            pks.append(PartitionKey(key=k, type_=t))
        return PartitionKeys(pks=pks)

    def make(self, *obj_arg: Union[SyftObject, Tuple[Any, ...]]) -> QueryKeys:
        if isinstance(obj_arg, SyftObject):
            return self.with_obj(obj_arg)
        else:
            return self.with_tuple(*obj_arg)


@serializable(recursive_serde=True)
class QueryKey(PartitionKey):
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
    def partition_key(self) -> PartitionKey:
        return PartitionKey(key=self.key, type_=self.type_)

    @staticmethod
    def from_obj(partition_key: PartitionKey, obj: SyftObject) -> List[Any]:
        pk_key = partition_key.key
        pk_type = partition_key.type_

        if isinstance(obj, pk_type):
            pk_value = obj
        else:
            pk_value = getattr(obj, pk_key)

        if pk_value and not isinstance(pk_value, pk_type):
            raise Exception(
                f"PartitionKey {pk_value} of type {type(pk_value)} must be {pk_type}."
            )
        return QueryKey(key=pk_key, type_=pk_type, value=pk_value)


@serializable(recursive_serde=True)
class PartitionKeysWithUID(PartitionKeys):
    uid_pk: PartitionKey

    @property
    def all(self) -> Iterable[PartitionKey]:
        all_keys = self.pks if isinstance(self.pks, (tuple, list)) else [self.pks]
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
    def from_obj(partition_keys: PartitionKeys, obj: SyftObject) -> QueryKeys:
        qks = []
        for partition_key in partition_keys.all:
            pk_key = partition_key.key
            pk_type = partition_key.type_
            pk_value = getattr(obj, pk_key)
            if pk_value and not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PartitionKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
            qk = QueryKey(key=pk_key, type_=pk_type, value=pk_value)
            qks.append(qk)
        return QueryKeys(qks=qks)

    @staticmethod
    def from_tuple(partition_keys: PartitionKeys, args: Tuple[Any, ...]) -> QueryKeys:
        qks = []
        for partition_key, pk_value in zip(partition_keys.all, args):
            pk_key = partition_key.key
            pk_type = partition_key.type_
            if not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PartitionKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
            qk = QueryKey(key=pk_key, type_=pk_type, value=pk_value)
            qks.append(qk)
        return QueryKeys(qks=qks)

    @staticmethod
    def from_dict(qks_dict: Dict[str, Any]) -> QueryKeys:
        qks = []
        for k, v in qks_dict.items():
            qks.append(QueryKey(key=k, type_=type(v), value=v))
        return QueryKeys(qks=qks)


UIDPartitionKey = PartitionKey(key="id", type_=UID)


@serializable(recursive_serde=True)
class PartitionSettings(SyftBaseModel):
    name: str
    object_type: type
    store_key: PartitionKey = UIDPartitionKey
    db_name: str

    @property
    def unique_keys(self) -> PartitionKeys:
        unique_keys = PartitionKeys.from_dict(self.object_type._syft_unique_keys_dict())
        return unique_keys.add(self.store_key)

    @property
    def searchable_keys(self) -> PartitionKeys:
        return PartitionKeys.from_dict(self.object_type._syft_searchable_keys_dict())


@serializable(recursive_serde=True)
class UniqueKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


@instrument
@serializable(recursive_serde=True)
class StorePartition:
    def __init__(self, settings: PartitionSettings) -> None:
        self.settings = settings
        self.init_store()

    def init_store(self) -> None:
        self.unique_cks = self.settings.unique_keys.all
        self.searchable_cks = self.settings.searchable_keys.all

    def store_query_key(self, obj: Any) -> QueryKey:
        return self.settings.store_key.with_obj(obj)

    def store_query_keys(self, objs: Any) -> QueryKeys:
        return QueryKeys(qks=[self.store_query_key(obj) for obj in objs])

    def find_index_or_search_keys(self, index_qks: QueryKeys, search_qks: QueryKeys):
        raise NotImplementedError

    def set(self, obj: SyftObject) -> Result[SyftObject, str]:
        raise NotImplementedError

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        raise NotImplementedError

    def get_all_from_store(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        raise NotImplementedError

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        raise NotImplementedError

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        raise NotImplementedError


@instrument
@serializable(recursive_serde=True)
class DocumentStore:
    partitions: Dict[str, StorePartition]
    partition_type: Type[StorePartition]

    def __init__(self) -> None:
        self.partitions = {}

    def partition(self, settings: PartitionSettings) -> StorePartition:
        if settings.name not in self.partitions:
            self.partitions[settings.name] = self.partition_type(settings=settings)
        return self.partitions[settings.name]


@instrument
class BaseStash:
    object_type: Type[SyftObject]
    settings: PartitionSettings
    partition: StorePartition

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.partition = store.partition(type(self).settings)

    def set(self, obj: BaseStash.object_type) -> Result[BaseStash.object_type, str]:
        return self.partition.set(obj=obj)

    def query_all(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[List[BaseStash.object_type], str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=qks)

        unique_keys = []
        searchable_keys = []

        for qk in qks.all:
            pk = qk.partition_key
            if pk in self.partition.unique_cks:
                unique_keys.append(qk)
            elif pk in self.partition.searchable_cks:
                searchable_keys.append(qk)
            else:
                return Err(
                    f"{qk} not in {type(self.partition)} unique or searchable keys"
                )

        index_qks = QueryKeys(qks=unique_keys)
        search_qks = QueryKeys(qks=searchable_keys)

        return self.partition.find_index_or_search_keys(
            index_qks=index_qks, search_qks=search_qks
        )

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

    def find_and_delete(self, **kwargs: Dict[str, Any]) -> Result[SyftSuccess, Err]:
        obj = self.query_one_kwargs(**kwargs)
        if obj.is_err():
            return obj.err()
        else:
            obj = obj.ok()

        if not obj:
            return Err(f"Object does not exists with kwargs: {kwargs}")
        qk = self.partition.store_query_key(obj)
        return self.delete(qk=qk)

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        return self.partition.delete(qk=qk)

    def update(
        self, obj: BaseStash.object_type
    ) -> Optional[Result[BaseStash.object_type, str]]:

        qk = self.partition.store_query_key(obj)
        return self.partition.update(qk=qk, obj=obj)

        return Err(
            f"Invalid Query Key Type. "
            f"Required: {self.partition.settings.store_key}, Found: {qk.partition_key}"
        )


# 🟡 TODO 26: the base partition is already a dict partition but we can change it later
@serializable(recursive_serde=True)
class DictStorePartition(StorePartition):
    def __init__(self, settings: PartitionSettings) -> None:
        self.data = {}
        super().__init__(settings=settings)

    def init_store(self) -> None:
        super().init_store()

        self.unique_keys = {}

        for partition_key in self.unique_cks:
            pk_key = partition_key.key
            self.unique_keys[pk_key] = {}

        self.searchable_keys = {}
        for partition_key in self.searchable_cks:
            pk_key = partition_key.key
            self.searchable_keys[pk_key] = defaultdict(list)

    def validate_partition_keys(
        self, store_query_key: QueryKey, unique_query_keys: QueryKeys
    ) -> UniqueKeyCheck:
        matches = []
        qks = unique_query_keys.all
        for qk in qks:
            pk_key, pk_value = qk.key, qk.value
            if pk_key not in self.unique_keys:
                raise Exception(
                    f"pk_key: {pk_key} not in unique_keys: {self.unique_keys.keys()}"
                )
            ck_col = self.unique_keys[pk_key]
            if pk_value in ck_col and ck_col[pk_value] == store_query_key.value:
                matches.append(pk_key)

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
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.unique_keys[pk_key]
            ck_col[pk_value] = store_query_key.value
            self.unique_keys[pk_key] = ck_col

        self.unique_keys[store_query_key.key][
            store_query_key.value
        ] = store_query_key.value

        sqks = searchable_query_keys.all
        for qk in sqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.searchable_keys[pk_key]
            ck_col[pk_value].append(store_query_key.value)
            self.searchable_keys[pk_key] = ck_col

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

            ck_check = self.validate_partition_keys(
                store_query_key=store_query_key, unique_query_keys=unique_query_keys
            )
            if not exists and ck_check == UniqueKeyCheck.EMPTY:
                self.set_data_and_keys(
                    store_query_key=store_query_key,
                    unique_query_keys=unique_query_keys,
                    searchable_query_keys=searchable_query_keys,
                    obj=obj,
                )
        except Exception as e:
            return Err(f"Failed to write obj {obj}. {e}")
        return Ok(obj)

    def find_index_or_search_keys(
        self, index_qks: QueryKeys, search_qks: QueryKeys
    ) -> Result[List[SyftObject], str]:

        ids: Optional[Set] = None
        errors = []
        if len(index_qks.all) > 0:
            index_results = self._get_keys_index(qks=index_qks)
            if index_results.is_ok():
                if ids is None:
                    ids = index_results.ok()
                ids = ids.intersection(index_results.ok())
            else:
                errors.append(index_results.err())

        search_results = None
        if len(search_qks.all) > 0:
            search_results = self._find_keys_search(qks=QueryKeys(qks=search_qks))

            if search_results.is_ok():
                if ids is None:
                    ids = search_results.ok()
                ids = ids.intersection(search_results.ok())
            else:
                errors.append(search_results.err())

        if len(errors) > 0:
            return Err(" ".join(errors))

        qks = self.store_query_keys(ids)
        return self.get_all_from_store(qks=qks)

    def remove_keys(
        self,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
    ) -> None:

        uqks = unique_query_keys.all
        for qk in uqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.unique_keys[pk_key]
            ck_col.pop(pk_value, None)

        sqks = searchable_query_keys.all
        for qk in sqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.searchable_keys[pk_key]
            ck_col.pop(pk_value, None)

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        try:
            if qk.value not in self.data:
                return Err(f"No object exists for query key: {qk}")

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
            for key, value in obj.dict(exclude_none=True).items():
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

    def _delete_unique_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _unique_ck in self.unique_cks:
            qk = _unique_ck.with_obj(obj)
            self.unique_keys[qk.key].pop(qk.value, None)
        return Ok(SyftSuccess(message="Deleted"))

    def _delete_search_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _search_ck in self.searchable_cks:
            qk = _search_ck.with_obj(obj)
            self.searchable_keys[qk.key].pop(qk.value, None)
        return Ok(SyftSuccess(message="Deleted"))

    def _get_keys_index(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.unique_keys:
                    return Err(f"Failed to query index with {qk}")
                ck_col = self.unique_keys[pk_key]
                if pk_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_value = ck_col[pk_value]
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

    def _find_keys_search(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset = {}
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.searchable_keys:
                    return Err(f"Failed to search with {qk}")
                ck_col = self.searchable_keys[pk_key]
                if pk_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_values = ck_col[pk_value]
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

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        try:
            _obj = self.data.pop(qk.value)
            self._delete_unique_keys_for(_obj)
            self._delete_search_keys_for(_obj)
            return Ok(SyftSuccess(message="Deleted"))
        except Exception as e:
            return Err(f"Failed to delete with query key {qk} with error: {e}")


# the base document store is already a dict but we can change it later
@serializable(recursive_serde=True)
class DictDocumentStore(DocumentStore):
    partition_type = DictStorePartition
