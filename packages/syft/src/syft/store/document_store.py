# future
from __future__ import annotations

# stdlib
import sys
import types
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

# third party
from pydantic import BaseModel
from result import Err
from result import Ok
from result import Result
from typeguard import check_type

# relative
from ..node.credentials import SyftSigningKey
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from ..service.action.action_permissions import ActionObjectPermission
from ..service.context import AuthedServiceContext
from ..service.response import SyftSuccess
from ..types.base import SyftBaseModel
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.telemetry import instrument
from .locks import LockingConfig
from .locks import NoLockingConfig
from .locks import SyftLock


@serializable()
class BasePartitionSettings(SyftBaseModel):
    """Basic Partition Settings

    Parameters:
        name: str
            Identifier to be used as prefix by stores and for partitioning
    """

    name: str


def first_or_none(result: Any) -> Ok:
    if hasattr(result, "__len__") and len(result) > 0:
        return Ok(result[0])
    return Ok(None)


if sys.version_info >= (3, 9):

    def is_generic_alias(t: type):
        return isinstance(t, (types.GenericAlias, typing._GenericAlias))

else:

    def is_generic_alias(t: type):
        return isinstance(t, typing._GenericAlias)


class StoreClientConfig(BaseModel):
    """Base Client specific configuration"""

    pass


@serializable()
class PartitionKey(BaseModel):
    key: str
    type_: Union[type, object]

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) == type(self)
            and self.key == other.key
            and self.type_ == other.type_
        )

    def with_obj(self, obj: Any) -> QueryKey:
        return QueryKey.from_obj(partition_key=self, obj=obj)

    def extract_list(self, obj: Any) -> List:
        # not a list and matches the internal list type of the _GenericAlias
        if not isinstance(obj, list):
            if not isinstance(obj, typing.get_args(self.type_)):
                obj = getattr(obj, self.key)
                if isinstance(obj, (types.FunctionType, types.MethodType)):
                    obj = obj()

            if not isinstance(obj, list) and isinstance(
                obj, typing.get_args(self.type_)
            ):
                # still not a list but the right type
                obj = [obj]

        # is a list type so lets compare directly
        check_type("obj", obj, self.type_)
        return obj

    @property
    def type_list(self) -> bool:
        return is_generic_alias(self.type_) and self.type_.__origin__ == list


@serializable()
class PartitionKeys(BaseModel):
    pks: Union[PartitionKey, Tuple[PartitionKey, ...], List[PartitionKey]]

    @property
    def all(self) -> List[PartitionKey]:
        # make sure we always return a list even if there's a single value
        return self.pks if isinstance(self.pks, (tuple, list)) else [self.pks]

    def with_obj(self, obj: Any) -> QueryKeys:
        return QueryKeys.from_obj(partition_keys=self, obj=obj)

    def with_tuple(self, *args: Any) -> QueryKeys:
        return QueryKeys.from_tuple(partition_keys=self, args=args)

    def add(self, pk: PartitionKey) -> PartitionKeys:
        return PartitionKeys(pks=list(self.all) + [pk])

    @staticmethod
    def from_dict(cks_dict: Dict[str, type]) -> PartitionKeys:
        pks = []
        for k, t in cks_dict.items():
            pks.append(PartitionKey(key=k, type_=t))
        return PartitionKeys(pks=pks)


@serializable()
class QueryKey(PartitionKey):
    value: Any

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) == type(self)
            and self.key == other.key
            and self.type_ == other.type_
            and self.value == other.value
        )

    @property
    def partition_key(self) -> PartitionKey:
        return PartitionKey(key=self.key, type_=self.type_)

    @staticmethod
    def from_obj(partition_key: PartitionKey, obj: Any) -> QueryKey:
        pk_key = partition_key.key
        pk_type = partition_key.type_

        # 🟡 TODO: support more advanced types than List[type]
        if partition_key.type_list:
            pk_value = partition_key.extract_list(obj)
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


@serializable()
class PartitionKeysWithUID(PartitionKeys):
    uid_pk: PartitionKey

    @property
    def all(self) -> List[PartitionKey]:
        all_keys = self.pks if isinstance(self.pks, (tuple, list)) else [self.pks]
        if self.uid_pk not in all_keys:
            all_keys.insert(0, self.uid_pk)
        return all_keys


@serializable()
class QueryKeys(SyftBaseModel):
    qks: Union[QueryKey, Tuple[QueryKey, ...], List[QueryKey]]

    @property
    def all(self) -> List[QueryKey]:
        # make sure we always return a list even if there's a single value
        return self.qks if isinstance(self.qks, (tuple, list)) else [self.qks]

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
                pk_value = partition_key.extract_list(obj)
            else:
                if pk_value and not isinstance(pk_value, pk_type):
                    raise Exception(
                        f"PartitionKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                    )
            qk = QueryKey(key=pk_key, type_=pk_type, value=pk_value)
            qks.append(qk)
        return QueryKeys(qks=qks)

    @staticmethod
    def from_tuple(partition_keys: PartitionKeys, args: Tuple) -> QueryKeys:
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


@serializable()
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
@serializable(attrs=["settings", "store_config", "unique_cks", "searchable_cks"])
class StorePartition:
    """Base StorePartition

    Parameters:
        settings: PartitionSettings
            PySyft specific settings
        store_config: StoreConfig
            Backend specific configuration
    """

    def __init__(
        self,
        root_verify_key: SyftVerifyKey,
        settings: PartitionSettings,
        store_config: StoreConfig,
    ) -> None:
        if root_verify_key is None:
            root_verify_key = SyftSigningKey.generate().verify_key
        self.root_verify_key = root_verify_key
        self.settings = settings
        self.store_config = store_config
        self.init_store()

        store_config.locking_config.lock_name = settings.name
        self.lock = SyftLock(store_config.locking_config)

    def init_store(self) -> Result[Ok, Err]:
        try:
            self.unique_cks = self.settings.unique_keys.all
            self.searchable_cks = self.settings.searchable_keys.all
        except BaseException as e:
            return Err(str(e))

        return Ok()

    def matches_unique_cks(self, partition_key: PartitionKey) -> bool:
        return partition_key in self.unique_cks

    def matches_searchable_cks(self, partition_key: PartitionKey) -> bool:
        return partition_key in self.searchable_cks

    def store_query_key(self, obj: Any) -> QueryKey:
        return self.settings.store_key.with_obj(obj)

    def store_query_keys(self, objs: Any) -> QueryKeys:
        return QueryKeys(qks=[self.store_query_key(obj) for obj in objs])

    # Thread-safe methods
    def _thread_safe_cbk(self, cbk: Callable, *args, **kwargs):
        locked = self.lock.acquire(blocking=True)
        if not locked:
            print("FAILED TO LOCK")
            return Err("Failed to acquire lock for the operation")

        try:
            result = cbk(*args, **kwargs)
        except BaseException as e:
            result = Err(str(e))
        self.lock.release()

        return result

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        return self._thread_safe_cbk(
            self._set,
            credentials=credentials,
            obj=obj,
            add_permissions=add_permissions,
            ignore_duplicates=ignore_duplicates,
        )

    def get(
        self,
        credentials: SyftVerifyKey,
        uid: UID,
    ) -> Result[SyftObject, str]:
        return self._thread_safe_cbk(
            self._get,
            uid=uid,
            credentials=credentials,
        )

    def find_index_or_search_keys(
        self,
        credentials: SyftVerifyKey,
        index_qks: QueryKeys,
        search_qks: QueryKeys,
        order_by: Optional[PartitionKey] = None,
    ) -> Result[List[SyftObject], str]:
        return self._thread_safe_cbk(
            self._find_index_or_search_keys,
            credentials,
            index_qks=index_qks,
            search_qks=search_qks,
            order_by=order_by,
        )

    def remove_keys(
        self,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
    ) -> None:
        self._thread_safe_cbk(
            self._remove_keys,
            unique_query_keys=unique_query_keys,
            searchable_query_keys=searchable_query_keys,
        )

    def update(
        self,
        credentials: SyftVerifyKey,
        qk: QueryKey,
        obj: SyftObject,
        has_permission=False,
    ) -> Result[SyftObject, str]:
        return self._thread_safe_cbk(
            self._update,
            credentials=credentials,
            qk=qk,
            obj=obj,
            has_permission=has_permission,
        )

    def get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: Optional[PartitionKey] = None,
    ) -> Result[List[SyftObject], str]:
        return self._thread_safe_cbk(
            self._get_all_from_store, credentials, qks, order_by
        )

    def delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission=False
    ) -> Result[SyftSuccess, Err]:
        return self._thread_safe_cbk(
            self._delete, credentials, qk, has_permission=has_permission
        )

    def all(
        self,
        credentials: SyftVerifyKey,
        order_by: Optional[PartitionKey] = None,
        has_permission: Optional[bool] = False,
    ) -> Result[List[BaseStash.object_type], str]:
        return self._thread_safe_cbk(self._all, credentials, order_by, has_permission)

    def migrate_data(
        self,
        to_klass: SyftObject,
        context: AuthedServiceContext,
        has_permission: Optional[bool] = False,
    ) -> Result[bool, str]:
        return self._thread_safe_cbk(
            self._migrate_data, to_klass, context, has_permission
        )

    # Potentially thread-unsafe methods.
    # CAUTION:
    #       * Don't use self.lock here.
    #       * Do not call the public thread-safe methods here(with locking).
    # These methods are called from the public thread-safe API, and will hang the process.
    def _set(
        self,
        obj: SyftObject,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        raise NotImplementedError

    def _update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        raise NotImplementedError

    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: Optional[PartitionKey] = None,
    ) -> Result[List[SyftObject], str]:
        raise NotImplementedError

    def _delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        raise NotImplementedError

    def _all(self) -> Result[List[BaseStash.object_type], str]:
        raise NotImplementedError

    def add_permission(self, permission: ActionObjectPermission) -> None:
        raise NotImplementedError

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        raise NotImplementedError

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        raise NotImplementedError

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        raise NotImplementedError

    def _migrate_data(
        self,
        to_klass: SyftObject,
        context: AuthedServiceContext,
        has_permission: bool,
    ) -> Result[bool, str]:
        raise NotImplementedError


@instrument
@serializable()
class DocumentStore:
    """Base Document Store

    Parameters:
        store_config: StoreConfig
            Store specific configuration.
    """

    partitions: Dict[str, StorePartition]
    partition_type: Type[StorePartition]

    def __init__(
        self, root_verify_key: Optional[SyftVerifyKey], store_config: StoreConfig
    ) -> None:
        if store_config is None:
            raise Exception("must have store config")
        self.partitions = {}
        self.store_config = store_config
        self.root_verify_key = root_verify_key

    def partition(self, settings: PartitionSettings) -> StorePartition:
        if settings.name not in self.partitions:
            self.partitions[settings.name] = self.partition_type(
                root_verify_key=self.root_verify_key,
                settings=settings,
                store_config=self.store_config,
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

    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: Optional[PartitionKey] = None,
        has_permission: bool = False,
    ) -> Result[List[BaseStash.object_type], str]:
        return self.partition.all(credentials, order_by, has_permission)

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        self.partition.add_permissions(permissions)

    def add_permission(self, permission: ActionObjectPermission) -> None:
        self.partition.add_permission(permission)

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        self.partition.remove_permission(permission)

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        return self.partition.has_permission(permission=permission)

    def __len__(self) -> int:
        return len(self.partition)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: BaseStash.object_type,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[BaseStash.object_type, str]:
        return self.partition.set(
            credentials=credentials,
            obj=obj,
            ignore_duplicates=ignore_duplicates,
            add_permissions=add_permissions,
        )

    def query_all(
        self,
        credentials: SyftVerifyKey,
        qks: Union[QueryKey, QueryKeys],
        order_by: Optional[PartitionKey] = None,
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
            credentials=credentials,
            index_qks=index_qks,
            search_qks=search_qks,
            order_by=order_by,
        )

    def query_all_kwargs(
        self,
        credentials: SyftVerifyKey,
        **kwargs: Dict[str, Any],
    ) -> Result[List[BaseStash.object_type], str]:
        order_by = kwargs.pop("order_by", None)
        qks = QueryKeys.from_dict(kwargs)
        return self.query_all(credentials=credentials, qks=qks, order_by=order_by)

    def query_one(
        self,
        credentials: SyftVerifyKey,
        qks: Union[QueryKey, QueryKeys],
        order_by: Optional[PartitionKey] = None,
    ) -> Result[Optional[BaseStash.object_type], str]:
        return self.query_all(
            credentials=credentials, qks=qks, order_by=order_by
        ).and_then(first_or_none)

    def query_one_kwargs(
        self,
        credentials: SyftVerifyKey,
        **kwargs: Dict[str, Any],
    ) -> Result[Optional[BaseStash.object_type], str]:
        return self.query_all_kwargs(credentials, **kwargs).and_then(first_or_none)

    def find_all(
        self, credentials: SyftVerifyKey, **kwargs: Dict[str, Any]
    ) -> Result[List[BaseStash.object_type], str]:
        return self.query_all_kwargs(credentials=credentials, **kwargs)

    def find_one(
        self, credentials: SyftVerifyKey, **kwargs: Dict[str, Any]
    ) -> Result[Optional[BaseStash.object_type], str]:
        return self.query_one_kwargs(credentials=credentials, **kwargs)

    def find_and_delete(
        self, credentials: SyftVerifyKey, **kwargs: Dict[str, Any]
    ) -> Result[SyftSuccess, Err]:
        obj = self.query_one_kwargs(credentials=credentials, **kwargs)
        if obj.is_err():
            return obj
        else:
            obj = obj.ok()

        if not obj:
            return Err(f"Object does not exists with kwargs: {kwargs}")
        qk = self.partition.store_query_key(obj)
        return self.delete(credentials=credentials, qk=qk)

    def delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission=False
    ) -> Result[SyftSuccess, Err]:
        return self.partition.delete(
            credentials=credentials, qk=qk, has_permission=has_permission
        )

    def update(
        self,
        credentials: SyftVerifyKey,
        obj: BaseStash.object_type,
        has_permission=False,
    ) -> Result[BaseStash.object_type, str]:
        qk = self.partition.store_query_key(obj)
        return self.partition.update(
            credentials=credentials, qk=qk, obj=obj, has_permission=has_permission
        )


@instrument
class BaseUIDStoreStash(BaseStash):
    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(credentials=credentials, qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[BaseUIDStoreStash.object_type], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(credentials=credentials, qks=qks)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: BaseUIDStoreStash.object_type,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[BaseUIDStoreStash.object_type, str]:
        res = self.check_type(obj, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(
            credentials=credentials,
            obj=res.ok(),
            ignore_duplicates=ignore_duplicates,
            add_permissions=add_permissions,
        )


@serializable()
class StoreConfig(SyftBaseObject):
    """Base Store configuration

    Parameters:
        store_type: Type
            Document Store type
        client_config: Optional[StoreClientConfig]
            Backend-specific config
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.
                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.
            Defaults to NoLockingConfig.
    """

    __canonical_name__ = "StoreConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    store_type: Type[DocumentStore]
    client_config: Optional[StoreClientConfig]
    locking_config: LockingConfig = NoLockingConfig()
