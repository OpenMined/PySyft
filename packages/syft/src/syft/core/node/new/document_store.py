# future
from __future__ import annotations

# stdlib
from functools import partial
import types
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import _GenericAlias

# third party
from pydantic import BaseModel
from result import Err
from result import Ok
from result import Result
from typeguard import check_type

# relative
from ....telemetry import instrument
from .base import SyftBaseModel
from .response import SyftSuccess
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftBaseObject
from .syft_object import SyftObject
from .uid import UID


@serializable(recursive_serde=True)
class BasePartitionSettings(SyftBaseModel):
    name: str


def first_or_none(result: Any) -> Optional[Any]:
    if hasattr(result, "__len__") and len(result) > 0:
        return Ok(result[0])
    return Ok(None)


class StoreClientConfig(BaseModel):
    pass


@serializable(recursive_serde=True)
class PartitionKey(BaseModel):
    key: str
    type_: Union[type, object]

    def __eq__(self, other: Any) -> bool:
        if type(other) == type(self):
            return self.key == other.key and self.type_ == other.type_
        return False

    def with_obj(self, obj: SyftObject) -> QueryKey:
        return QueryKey.from_obj(partition_key=self, obj=obj)

    def is_valid_list(self, obj: SyftObject) -> bool:
        # not a list and matches the internal list type of the _GenericAlias
        if not isinstance(obj, list):
            if not isinstance(obj, self.type_.__args__):
                obj = getattr(obj, self.key)
                if isinstance(obj, (types.FunctionType, types.MethodType)):
                    obj = obj()

            if not isinstance(obj, list) and isinstance(obj, self.type_.__args__):
                # still not a list but the right type
                obj = [obj]

        # is a list type so lets compare directly
        check_type("obj", obj, self.type_)
        return obj

    @property
    def type_list(self) -> bool:
        if isinstance(self.type_, _GenericAlias) and self.type_.__origin__ == list:
            return True
        return False


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

        # 🟡 TODO: support more advanced types than List[type]
        if partition_key.type_list:
            pk_value = partition_key.is_valid_list(obj)
        else:
            if isinstance(obj, pk_type):
                pk_value = obj
            else:
                pk_value = getattr(obj, pk_key)
                # object has a method for getting these types
                # we can't use properties because we don't seem to be able to get the
                # return types
                if isinstance(pk_value, (types.FunctionType, types.MethodType)):
                    pk_value = pk_value()

            if pk_value and not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PartitionKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
        return QueryKey(key=pk_key, type_=pk_type, value=pk_value)

    @property
    def as_dict(self):
        return {self.key: self.value}

    @property
    def as_dict_mongo(self):
        key = self.key
        if key == "id":
            key = "_id"
        if self.type_list:
            # We want to search inside the list of values
            return {key: {"$in": self.value}}
        return {key: self.value}


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
            # object has a method for getting these types
            # we can't use properties because we don't seem to be able to get the
            # return types
            if isinstance(pk_value, (types.FunctionType, types.MethodType)):
                pk_value = pk_value()
            if partition_key.type_list:
                pk_value = partition_key.is_valid_list(obj)
            else:
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

    @property
    def as_dict(self):
        qk_dict = {}
        for qk in self.all:
            qk_key = qk.key
            qk_value = qk.value
            qk_dict[qk_key] = qk_value
        return qk_dict

    @property
    def as_dict_mongo(self):
        qk_dict = {}
        for qk in self.all:
            qk_key = qk.key
            qk_value = qk.value
            if qk_key == "id":
                qk_key = "_id"
            if qk.type_list:
                # We want to search inside the list of values
                qk_dict[qk_key] = {"$in": qk_value}
            else:
                qk_dict[qk_key] = qk_value
        return qk_dict


UIDPartitionKey = PartitionKey(key="id", type_=UID)


@serializable(recursive_serde=True)
class PartitionSettings(BasePartitionSettings):
    object_type: type
    store_key: PartitionKey = UIDPartitionKey

    @property
    def unique_keys(self) -> PartitionKeys:
        unique_keys = PartitionKeys.from_dict(self.object_type._syft_unique_keys_dict())
        return unique_keys.add(self.store_key)

    @property
    def searchable_keys(self) -> PartitionKeys:
        return PartitionKeys.from_dict(self.object_type._syft_searchable_keys_dict())


@instrument
@serializable(recursive_serde=True)
class StorePartition:
    def __init__(
        self,
        settings: PartitionSettings,
        store_config: StoreConfig,
    ) -> None:
        self.settings = settings
        self.store_config = store_config
        self.init_store()

    def init_store(self) -> None:
        self.unique_cks = self.settings.unique_keys.all
        self.searchable_cks = self.settings.searchable_keys.all

    def matches_unique_cks(self, partition_key: PartitionKey) -> bool:
        if partition_key in self.unique_cks:
            return True
        return False

    def matches_searchable_cks(self, partition_key: PartitionKey) -> bool:
        if partition_key in self.searchable_cks:
            return True
        return False

    def store_query_key(self, obj: Any) -> QueryKey:
        return self.settings.store_key.with_obj(obj)

    def store_query_keys(self, objs: Any) -> QueryKeys:
        return QueryKeys(qks=[self.store_query_key(obj) for obj in objs])

    def find_index_or_search_keys(self, index_qks: QueryKeys, search_qks: QueryKeys):
        raise NotImplementedError

    def all(self) -> Result[List[BaseStash.object_type], str]:
        raise NotImplementedError

    def set(
        self,
        obj: SyftObject,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
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

    def __init__(self, store_config: StoreConfig) -> None:
        if store_config is None:
            raise Exception("must have store config")
        self.partitions = {}
        self.store_config = store_config

    def partition(self, settings: PartitionSettings) -> StorePartition:
        if settings.name not in self.partitions:
            self.partitions[settings.name] = self.partition_type(
                settings=settings, store_config=self.store_config
            )
        return self.partitions[settings.name]


@instrument
class BaseStash:
    object_type: Type[SyftObject]
    settings: PartitionSettings
    partition: StorePartition

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.partition = store.partition(type(self).settings)

    def check_type(self, obj: Any, type_: type) -> Result[Any, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    def get_all(self) -> Result[List[BaseStash.object_type], str]:
        return self.partition.all()

    def __len__(self) -> int:
        return self.partition.__len__()

    def set(
        self,
        obj: BaseStash.object_type,
        ignore_duplicates: bool = False,
    ) -> Result[BaseStash.object_type, str]:
        return self.partition.set(obj=obj, ignore_duplicates=ignore_duplicates)

    def query_all(
        self, qks: Union[QueryKey, QueryKeys]
    ) -> Result[List[BaseStash.object_type], str]:
        if isinstance(qks, QueryKey):
            qks = QueryKeys(qks=qks)

        unique_keys = []
        searchable_keys = []

        for qk in qks.all:
            pk = qk.partition_key
            if self.partition.matches_unique_cks(pk):
                unique_keys.append(qk)
            elif self.partition.matches_searchable_cks(pk):
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


@instrument
class BaseUIDStoreStash(BaseStash):
    def delete_by_uid(self, uid: UID) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result.err()

    def get_by_uid(
        self, uid: UID
    ) -> Result[Optional[BaseUIDStoreStash.object_type], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(qks=qks)

    def set(
        self,
        obj: BaseUIDStoreStash.object_type,
        ignore_duplicates: bool = False,
    ) -> Result[BaseUIDStoreStash.object_type, str]:
        set_method = partial(super().set, ignore_duplicates=ignore_duplicates)
        return self.check_type(obj, self.object_type).and_then(set_method)


@serializable(recursive_serde=True)
class StoreConfig(SyftBaseObject):
    __canonical_name__ = "StoreConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    store_type: Type[DocumentStore]
    client_config: Optional[StoreClientConfig]
