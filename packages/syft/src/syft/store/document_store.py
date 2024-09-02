# future
from __future__ import annotations

# stdlib
from collections.abc import Callable
import types
import typing
from typing import Any
from typing import Literal
from typing import TypeVar

# third party
from pydantic import BaseModel
from pydantic import Field
from typeguard import check_type

# relative
from ..serde.serializable import serializable
from ..server.credentials import SyftSigningKey
from ..server.credentials import SyftVerifyKey
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import StoragePermission
from ..service.context import AuthedServiceContext
from ..service.response import SyftSuccess
from ..types.base import SyftBaseModel
from ..types.errors import SyftException
from ..types.result import Ok
from ..types.result import as_result
from ..types.syft_object import BaseDateTime
from ..types.syft_object import PartialSyftObject
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.uid import UID
from ..util.telemetry import instrument
from .document_store_errors import NotFoundException
from .document_store_errors import StashException
from .locks import LockingConfig
from .locks import NoLockingConfig
from .locks import SyftLock


@serializable(canonical_name="BasePartitionSettings", version=1)
class BasePartitionSettings(SyftBaseModel):
    """Basic Partition Settings

    Parameters:
        name: str
            Identifier to be used as prefix by stores and for partitioning
    """

    name: str


T = TypeVar("T")


def new_first_or_none(result: list[T]) -> T | None:
    if hasattr(result, "__len__") and len(result) > 0:
        return result[0]
    return None


# todo: remove
def first_or_none(result: Any) -> Ok:
    if hasattr(result, "__len__") and len(result) > 0:
        return Ok(result[0])
    return Ok(None)


def is_generic_alias(t: type) -> bool:
    return isinstance(t, types.GenericAlias | typing._GenericAlias)


class StoreClientConfig(BaseModel):
    """Base Client specific configuration"""

    pass


@serializable(canonical_name="PartitionKey", version=1)
class PartitionKey(BaseModel):
    key: str
    type_: type | object

    def __eq__(self, other: Any) -> bool:
        return (
            type(other) == type(self)
            and self.key == other.key
            and self.type_ == other.type_
        )

    def with_obj(self, obj: Any) -> QueryKey:
        return QueryKey.from_obj(partition_key=self, obj=obj)

    def extract_list(self, obj: Any) -> list:
        # not a list and matches the internal list type of the _GenericAlias
        if not isinstance(obj, list):
            if not isinstance(obj, typing.get_args(self.type_)):
                obj = getattr(obj, self.key)
                if isinstance(obj, types.FunctionType | types.MethodType):
                    obj = obj()

            if not isinstance(obj, list) and isinstance(
                obj, typing.get_args(self.type_)
            ):
                # still not a list but the right type
                obj = [obj]

        # is a list type so lets compare directly
        check_type(obj, self.type_)
        return obj

    @property
    def type_list(self) -> bool:
        return is_generic_alias(self.type_) and self.type_.__origin__ == list


@serializable(canonical_name="PartitionKeys", version=1)
class PartitionKeys(BaseModel):
    pks: PartitionKey | tuple[PartitionKey, ...] | list[PartitionKey]

    @property
    def all(self) -> tuple[PartitionKey, ...] | list[PartitionKey]:
        # make sure we always return a list even if there's a single value
        return self.pks if isinstance(self.pks, tuple | list) else [self.pks]

    def with_obj(self, obj: Any) -> QueryKeys:
        return QueryKeys.from_obj(partition_keys=self, obj=obj)

    def with_tuple(self, *args: Any) -> QueryKeys:
        return QueryKeys.from_tuple(partition_keys=self, args=args)

    def add(self, pk: PartitionKey) -> PartitionKeys:
        return PartitionKeys(pks=list(self.all) + [pk])

    @staticmethod
    def from_dict(cks_dict: dict[str, type]) -> PartitionKeys:
        pks = []
        for k, t in cks_dict.items():
            pks.append(PartitionKey(key=k, type_=t))
        return PartitionKeys(pks=pks)


@serializable(canonical_name="QueryKey", version=1)
class QueryKey(PartitionKey):
    value: Any = None

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
                # TODO: fix the mypy issue
                if isinstance(pk_value, types.FunctionType | types.MethodType):  # type: ignore[unreachable]
                    pk_value = pk_value()  # type: ignore[unreachable]

            if pk_value and not isinstance(pk_value, pk_type):
                raise Exception(
                    f"PartitionKey {pk_value} of type {type(pk_value)} must be {pk_type}."
                )
        return QueryKey(key=pk_key, type_=pk_type, value=pk_value)

    @property
    def as_dict(self) -> dict[str, Any]:
        return {self.key: self.value}

    @property
    def as_dict_mongo(self) -> dict[str, Any]:
        key = self.key
        if key == "id":
            key = "_id"
        if self.type_list:
            # We want to search inside the list of values
            return {key: {"$in": self.value}}
        return {key: self.value}


@serializable(canonical_name="PartitionKeysWithUID", version=1)
class PartitionKeysWithUID(PartitionKeys):
    uid_pk: PartitionKey

    @property
    def all(self) -> tuple[PartitionKey, ...] | list[PartitionKey]:
        all_keys = list(self.pks) if isinstance(self.pks, tuple | list) else [self.pks]
        if self.uid_pk not in all_keys:
            all_keys.insert(0, self.uid_pk)
        return all_keys


@serializable(canonical_name="QueryKeys", version=1)
class QueryKeys(SyftBaseModel):
    qks: QueryKey | tuple[QueryKey, ...] | list[QueryKey]

    @property
    def all(self) -> tuple[QueryKey, ...] | list[QueryKey]:
        # make sure we always return a list even if there's a single value
        return self.qks if isinstance(self.qks, tuple | list) else [self.qks]

    @staticmethod
    def from_obj(partition_keys: PartitionKeys, obj: SyftObject) -> QueryKeys:
        qks = []
        for partition_key in partition_keys.all:
            pk_key = partition_key.key  # name of the attribute
            pk_type = partition_key.type_
            pk_value = getattr(obj, pk_key)
            # object has a method for getting these types
            # we can't use properties because we don't seem to be able to get the
            # return types
            if isinstance(pk_value, types.FunctionType | types.MethodType):
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
    def from_tuple(partition_keys: PartitionKeys, args: tuple) -> QueryKeys:
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
    def from_dict(qks_dict: dict[str, Any]) -> QueryKeys:
        qks = []
        for k, v in qks_dict.items():
            qks.append(QueryKey(key=k, type_=type(v), value=v))
        return QueryKeys(qks=qks)

    @property
    def as_dict(self) -> dict:
        qk_dict = {}
        for qk in self.all:
            qk_key = qk.key
            qk_value = qk.value
            qk_dict[qk_key] = qk_value
        return qk_dict

    @property
    def as_dict_mongo(self) -> dict:
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


@serializable(canonical_name="PartitionSettings", version=1)
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


@serializable(
    attrs=["settings", "store_config", "unique_cks", "searchable_cks"],
    canonical_name="StorePartition",
    version=1,
)
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
        server_uid: UID,
        root_verify_key: SyftVerifyKey | None,
        settings: PartitionSettings,
        store_config: StoreConfig,
        has_admin_permissions: Callable[[SyftVerifyKey], bool] | None = None,
    ) -> None:
        if root_verify_key is None:
            root_verify_key = SyftSigningKey.generate().verify_key
        self.server_uid = server_uid
        self.root_verify_key = root_verify_key
        self.settings = settings
        self.store_config = store_config
        self.has_admin_permissions = has_admin_permissions
        self.init_store().unwrap(
            public_message="Something went wrong initializing the store"
        )
        store_config.locking_config.lock_name = f"StorePartition-{settings.name}"
        self.lock = SyftLock(store_config.locking_config)

    @as_result(SyftException)
    def init_store(self) -> bool:
        try:
            self.unique_cks = self.settings.unique_keys.all
            self.searchable_cks = self.settings.searchable_keys.all
        except BaseException as e:
            raise SyftException.from_exception(e)
        return True

    def matches_unique_cks(self, partition_key: PartitionKey) -> bool:
        return partition_key in self.unique_cks

    def matches_searchable_cks(self, partition_key: PartitionKey) -> bool:
        return partition_key in self.searchable_cks

    def store_query_key(self, obj: Any) -> QueryKey:
        return self.settings.store_key.with_obj(obj)

    def store_query_keys(self, objs: Any) -> QueryKeys:
        return QueryKeys(qks=[self.store_query_key(obj) for obj in objs])

    # Thread-safe methods
    @as_result(SyftException)
    def _thread_safe_cbk(self, cbk: Callable, *args: Any, **kwargs: Any) -> Any:
        locked = self.lock.acquire(blocking=True)
        if not locked:
            raise SyftException(
                public_message=f"Failed to acquire lock for the operation {self.lock.lock_name} ({self.lock._lock})"
            )

        try:
            result = cbk(*args, **kwargs).unwrap()
        except BaseException as e:
            raise SyftException.from_exception(e)
        finally:
            self.lock.release()

        return result

    @as_result(SyftException)
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> SyftObject:
        if obj.created_date is None:
            obj.created_date = BaseDateTime.now()
        return self._thread_safe_cbk(
            self._set,
            credentials=credentials,
            obj=obj,
            add_permissions=add_permissions,
            add_storage_permission=add_storage_permission,
            ignore_duplicates=ignore_duplicates,
        ).unwrap()

    @as_result(SyftException)
    def get(
        self,
        credentials: SyftVerifyKey,
        uid: UID,
    ) -> SyftObject:
        return self._thread_safe_cbk(
            self._get,
            uid=uid,
            credentials=credentials,
        ).unwrap()

    @as_result(SyftException)
    def find_index_or_search_keys(
        self,
        credentials: SyftVerifyKey,
        index_qks: QueryKeys,
        search_qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[SyftObject]:
        return self._thread_safe_cbk(
            self._find_index_or_search_keys,
            credentials,
            index_qks=index_qks,
            search_qks=search_qks,
            order_by=order_by,
        ).unwrap()

    @as_result(SyftException)
    def remove_keys(
        self,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
    ) -> None:
        return self._thread_safe_cbk(
            self._remove_keys,
            unique_query_keys=unique_query_keys,
            searchable_query_keys=searchable_query_keys,
        ).unwrap()

    @as_result(SyftException)
    def update(
        self,
        credentials: SyftVerifyKey,
        qk: QueryKey,
        obj: SyftObject,
        has_permission: bool = False,
    ) -> SyftObject:
        return self._thread_safe_cbk(
            self._update,
            credentials=credentials,
            qk=qk,
            obj=obj,
            has_permission=has_permission,
        ).unwrap()

    @as_result(SyftException)
    def get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[SyftObject]:
        return self._thread_safe_cbk(
            self._get_all_from_store, credentials, qks, order_by
        ).unwrap()

    @as_result(SyftException)
    def delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> SyftSuccess:
        return self._thread_safe_cbk(
            self._delete, credentials, qk, has_permission=has_permission
        ).unwrap()

    @as_result(SyftException)
    def all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool | None = False,
    ) -> list[NewBaseStash.object_type]:
        return self._thread_safe_cbk(
            self._all, credentials, order_by, has_permission
        ).unwrap()

    @as_result(SyftException)
    def migrate_data(
        self,
        to_klass: SyftObject,
        context: AuthedServiceContext,
        has_permission: bool | None = False,
    ) -> bool:
        return self._thread_safe_cbk(
            self._migrate_data, to_klass, context, has_permission
        ).unwrap()

    # Potentially thread-unsafe methods.
    # CAUTION:
    #       * Don't use self.lock here.
    #       * Do not call the public thread-safe methods here(with locking).
    # These methods are called from the public thread-safe API, and will hang the process.
    @as_result(SyftException)
    def _set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> SyftObject:
        raise NotImplementedError

    @as_result(SyftException)
    def _update(
        self,
        credentials: SyftVerifyKey,
        qk: QueryKey,
        obj: SyftObject,
        has_permission: bool = False,
        overwrite: bool = False,
        allow_missing_keys: bool = False,
    ) -> SyftObject:
        raise NotImplementedError

    @as_result(SyftException)
    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[SyftObject]:
        raise NotImplementedError

    @as_result(SyftException)
    def _delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> SyftSuccess:
        raise NotImplementedError

    @as_result(SyftException)
    def _all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool | None = False,
    ) -> list[NewBaseStash.object_type]:
        raise NotImplementedError

    def add_permission(self, permission: ActionObjectPermission) -> None:
        raise NotImplementedError

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        raise NotImplementedError

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        raise NotImplementedError

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        raise NotImplementedError

    @as_result(SyftException)
    def get_all_permissions(self) -> dict[UID, set[str]]:
        raise NotImplementedError

    def _get_permissions_for_uid(self, uid: UID) -> set[str]:
        raise NotImplementedError

    def add_storage_permission(self, permission: StoragePermission) -> None:
        raise NotImplementedError

    def add_storage_permissions(self, permissions: list[StoragePermission]) -> None:
        raise NotImplementedError

    def remove_storage_permission(self, permission: StoragePermission) -> None:
        raise NotImplementedError

    def has_storage_permission(self, permission: StoragePermission | UID) -> bool:
        raise NotImplementedError

    def _get_storage_permissions_for_uid(self, uid: UID) -> set[UID]:
        raise NotImplementedError

    @as_result(SyftException)
    def get_all_storage_permissions(self) -> dict[UID, set[UID]]:
        raise NotImplementedError

    @as_result(SyftException)
    def _migrate_data(
        self,
        to_klass: SyftObject,
        context: AuthedServiceContext,
        has_permission: bool,
    ) -> bool:
        raise NotImplementedError


@serializable(canonical_name="DocumentStore", version=1)
class DocumentStore:
    """Base Document Store

    Parameters:
        store_config: StoreConfig
            Store specific configuration.
    """

    partitions: dict[str, StorePartition]
    partition_type: type[StorePartition]

    def __init__(
        self,
        server_uid: UID,
        root_verify_key: SyftVerifyKey | None,
        store_config: StoreConfig,
    ) -> None:
        if store_config is None:
            raise Exception("must have store config")
        self.partitions = {}
        self.store_config = store_config
        self.server_uid = server_uid
        self.root_verify_key = root_verify_key

    def __has_admin_permissions(
        self, settings: PartitionSettings
    ) -> Callable[[SyftVerifyKey], bool]:
        # relative
        from ..service.user.user import User
        from ..service.user.user_roles import ServiceRole
        from ..service.user.user_stash import UserStash

        # leave out UserStash to avoid recursion
        # TODO: pass the callback from BaseStash instead of DocumentStore
        # so that this works with UserStash after the sqlite thread fix is merged
        if settings.object_type is User:
            return lambda credentials: False

        user_stash = UserStash(store=self)

        def has_admin_permissions(credentials: SyftVerifyKey) -> bool:
            res = user_stash.get_by_verify_key(
                credentials=credentials,
                verify_key=credentials,
            )

            return (
                res.is_ok()
                and (user := res.ok()) is not None
                and user.role in (ServiceRole.DATA_OWNER, ServiceRole.ADMIN)
            )

        return has_admin_permissions

    def partition(self, settings: PartitionSettings) -> StorePartition:
        if settings.name not in self.partitions:
            self.partitions[settings.name] = self.partition_type(
                server_uid=self.server_uid,
                root_verify_key=self.root_verify_key,
                settings=settings,
                store_config=self.store_config,
                has_admin_permissions=self.__has_admin_permissions(settings),
            )
        return self.partitions[settings.name]

    def get_partition_object_types(self) -> list[type]:
        return [
            partition.settings.object_type for partition in self.partitions.values()
        ]


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
            Defaults to NoLockingConfig.
    """

    __canonical_name__ = "StoreConfig"
    __version__ = SYFT_OBJECT_VERSION_1

    store_type: type[DocumentStore]
    client_config: StoreClientConfig | None = None
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)


@instrument
class NewBaseStash:
    object_type: type[SyftObject]
    settings: PartitionSettings
    partition: StorePartition

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.partition = store.partition(type(self).settings)

    @as_result(StashException)
    def check_type(self, obj: Any, type_: type) -> Any:
        if not isinstance(obj, type_):
            raise StashException(f"{type(obj)} does not match required type: {type_}")
        return obj

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> list[NewBaseStash.object_type]:
        return self.partition.all(credentials, order_by, has_permission).unwrap()

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        self.partition.add_permissions(permissions)

    def add_permission(self, permission: ActionObjectPermission) -> None:
        self.partition.add_permission(permission)

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        self.partition.remove_permission(permission)

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        return self.partition.has_permission(permission=permission)

    def has_storage_permission(self, permission: StoragePermission) -> bool:
        return self.partition.has_storage_permission(permission=permission)

    def __len__(self) -> int:
        return len(self.partition)

    @as_result(StashException)
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: NewBaseStash.object_type,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> NewBaseStash.object_type:
        return self.partition.set(
            credentials=credentials,
            obj=obj,
            ignore_duplicates=ignore_duplicates,
            add_permissions=add_permissions,
            add_storage_permission=add_storage_permission,
        ).unwrap()

    @as_result(StashException)
    def query_all(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKey | QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[NewBaseStash.object_type]:
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
                raise StashException(
                    f"{qk} not in {type(self.partition)} unique or searchable keys"
                )

        index_qks = QueryKeys(qks=unique_keys)
        search_qks = QueryKeys(qks=searchable_keys)

        return self.partition.find_index_or_search_keys(
            credentials=credentials,
            index_qks=index_qks,
            search_qks=search_qks,
            order_by=order_by,
        ).unwrap()

    @as_result(StashException)
    def query_all_kwargs(
        self,
        credentials: SyftVerifyKey,
        **kwargs: dict[str, Any],
    ) -> list[NewBaseStash.object_type]:
        order_by = kwargs.pop("order_by", None)
        qks = QueryKeys.from_dict(kwargs)
        # TODO: Check order_by type...
        return self.query_all(
            credentials=credentials, qks=qks, order_by=order_by
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def query_one(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKey | QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> NewBaseStash.object_type:
        result = self.query_all(
            credentials=credentials, qks=qks, order_by=order_by
        ).unwrap()
        value = new_first_or_none(result)
        if value is None:
            keys = qks.all if isinstance(qks, QueryKeys) else [qks]
            keys_str = ", ".join(f"{x.key}: {x.value}" for x in keys)
            raise NotFoundException(
                public_message=f"Could not find {self.object_type} with {keys_str}"
            )
        return value

    @as_result(StashException, NotFoundException)
    def query_one_kwargs(
        self,
        credentials: SyftVerifyKey,
        **kwargs: dict[str, Any],
    ) -> NewBaseStash.object_type:
        result = self.query_all_kwargs(credentials, **kwargs).unwrap()
        value = new_first_or_none(result)
        if value is None:
            raise NotFoundException
        return value

    @as_result(StashException)
    def find_all(
        self, credentials: SyftVerifyKey, **kwargs: dict[str, Any]
    ) -> list[NewBaseStash.object_type]:
        return self.query_all_kwargs(credentials=credentials, **kwargs).unwrap()

    @as_result(StashException, NotFoundException)
    def find_one(
        self, credentials: SyftVerifyKey, **kwargs: dict[str, Any]
    ) -> NewBaseStash.object_type:
        return self.query_one_kwargs(credentials=credentials, **kwargs).unwrap()

    @as_result(StashException, NotFoundException)
    def find_and_delete(
        self, credentials: SyftVerifyKey, **kwargs: dict[str, Any]
    ) -> Literal[True]:
        obj = self.query_one_kwargs(credentials=credentials, **kwargs).unwrap()
        qk = self.partition.store_query_key(obj)
        return self.delete(credentials=credentials, qk=qk).unwrap()

    @as_result(StashException, SyftException)
    def delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> Literal[True]:
        # TODO: (error) Check return response
        return self.partition.delete(
            credentials=credentials, qk=qk, has_permission=has_permission
        ).unwrap()

    @as_result(StashException, SyftException)
    def update(
        self,
        credentials: SyftVerifyKey,
        obj: NewBaseStash.object_type,
        has_permission: bool = False,
    ) -> NewBaseStash.object_type:
        # TODO: See what breaks:
        # this is for when we pass an somelike like a UserUpdate obj
        if isinstance(obj, PartialSyftObject):
            current = self.find_one(credentials, id=obj.id).unwrap()
            obj.apply(to=current)
            obj = current

        obj = self.check_type(obj, self.object_type).unwrap()
        qk = self.partition.store_query_key(obj)
        return self.partition.update(
            credentials=credentials, qk=qk, obj=obj, has_permission=has_permission
        ).unwrap()


@instrument
class NewBaseUIDStoreStash(NewBaseStash):
    @as_result(SyftException, StashException)
    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID, has_permission: bool = False
    ) -> UID:
        qk = UIDPartitionKey.with_obj(uid)
        super().delete(
            credentials=credentials, qk=qk, has_permission=has_permission
        ).unwrap()
        return uid

    @as_result(SyftException, StashException, NotFoundException)
    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> NewBaseUIDStoreStash.object_type:
        # TODO: Could change to query_one, no?
        result = self.partition.get(credentials=credentials, uid=uid).unwrap()
        if result is None:
            raise NotFoundException(
                public_message=f"{self.object_type} with uid {uid} not found"
            )

        return result

    @as_result(SyftException, StashException)
    def set(  # type: ignore [override]
        self,
        credentials: SyftVerifyKey,
        obj: NewBaseUIDStoreStash.object_type,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> NewBaseUIDStoreStash.object_type:
        self.check_type(obj, self.object_type).unwrap()
        return (
            super()
            .set(
                credentials=credentials,
                obj=obj,
                ignore_duplicates=ignore_duplicates,
                add_permissions=add_permissions,
                add_storage_permission=add_storage_permission,
            )
            .unwrap(
                public_message=f"Failed to set {self.object_type} with uid {obj.id} not found"
            )
        )
