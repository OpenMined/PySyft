# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import cast

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..server.credentials import SyftVerifyKey
from ..service.action.action_permissions import ActionObjectEXECUTE
from ..service.action.action_permissions import ActionObjectOWNER
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import ActionObjectREAD
from ..service.action.action_permissions import ActionObjectWRITE
from ..service.action.action_permissions import ActionPermission
from ..service.action.action_permissions import StoragePermission
from ..service.context import AuthedServiceContext
from ..service.response import SyftSuccess
from ..types.errors import SyftException
from ..types.result import as_result
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .document_store import NewBaseStash
from .document_store import PartitionKey
from .document_store import PartitionKeys
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StorePartition


@serializable(canonical_name="UniqueKeyCheck", version=1)
class UniqueKeyCheck(Enum):
    EMPTY = 0
    MATCHES = 1
    ERROR = 2


class KeyValueBackingStore:
    """Key-Value store core logic."""

    def __setitem__(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    def __getitem__(self, key: Any) -> Self:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def copy(self) -> Self:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def keys(self) -> Any:
        raise NotImplementedError

    def values(self) -> Any:
        raise NotImplementedError

    def items(self) -> Any:
        raise NotImplementedError

    def pop(self, *args: Any) -> Self:
        raise NotImplementedError

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Any:
        raise NotImplementedError


class KeyValueStorePartition(StorePartition):
    """Key-Value StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings
        `store_config`: StoreConfig
            Backend specific configuration
    """

    @as_result(SyftException)
    def init_store(self) -> bool:
        super().init_store().unwrap()

        try:
            self.data = self.store_config.backing_store(
                "data", self.settings, self.store_config
            )
            self.unique_keys = self.store_config.backing_store(
                "unique_keys", self.settings, self.store_config
            )
            self.searchable_keys = self.store_config.backing_store(
                "searchable_keys", self.settings, self.store_config
            )

            # uid -> set['<uid>_permission']
            self.permissions: dict[UID, set[str]] = self.store_config.backing_store(
                "permissions", self.settings, self.store_config, ddtype=set
            )

            # uid -> set['<server_uid>']
            self.storage_permissions: dict[UID, set[UID]] = (
                self.store_config.backing_store(
                    "storage_permissions",
                    self.settings,
                    self.store_config,
                    ddtype=set,
                )
            )

            for partition_key in self.unique_cks:
                pk_key = partition_key.key
                if pk_key not in self.unique_keys:
                    self.unique_keys[pk_key] = {}

            for partition_key in self.searchable_cks:
                pk_key = partition_key.key
                if pk_key not in self.searchable_keys:
                    self.searchable_keys[pk_key] = defaultdict(list)
        except BaseException as e:
            raise SyftException.from_exception(e)

        return True

    def __len__(self) -> int:
        return len(self.data)

    @as_result(SyftException, KeyError)
    def _get(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        has_permission: bool | None = False,
    ) -> SyftObject:
        # relative
        from ..service.action.action_store import ActionObjectREAD

        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)

        if self.has_permission(read_permission) or has_permission:
            syft_object = self.data[uid]
            return syft_object
        raise SyftException(public_message=f"Permission: {read_permission} denied")

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
        store_query_key: QueryKey = self.settings.store_key.with_obj(obj)
        uid = store_query_key.value
        write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
        unique_query_keys: QueryKeys = self.settings.unique_keys.with_obj(obj)
        store_key_exists = store_query_key.value in self.data
        searchable_query_keys = self.settings.searchable_keys.with_obj(obj)

        ck_check = self._check_partition_keys_unique(
            unique_query_keys=unique_query_keys
        ).unwrap()

        can_write = self.has_permission(write_permission)

        if not store_key_exists and ck_check == UniqueKeyCheck.EMPTY:
            # attempt to claim it for writing
            can_write = self.take_ownership(uid=uid, credentials=credentials).unwrap()
        elif not ignore_duplicates:
            keys = ", ".join(f"`{key.key}`" for key in unique_query_keys.all)
            raise SyftException(
                public_message=f"Duplication Key Error for {obj}.\n"
                f"The fields that should be unique are {keys}."
            )
        else:
            # we are not throwing an error, because we are ignoring duplicates
            # we are also not writing though
            return obj

        if not can_write:
            raise SyftException(public_message=f"Permission: {write_permission} denied")

        self._set_data_and_keys(
            store_query_key=store_query_key,
            unique_query_keys=unique_query_keys,
            searchable_query_keys=searchable_query_keys,
            obj=obj,
        )
        self.data[uid] = obj

        # Add default permissions
        if uid not in self.permissions:
            self.permissions[uid] = set()

        self.add_permission(ActionObjectREAD(uid=uid, credentials=credentials))

        if add_permissions is not None:
            self.add_permissions(add_permissions)

        if uid not in self.storage_permissions:
            self.storage_permissions[uid] = set()

        if add_storage_permission:
            self.add_storage_permission(
                StoragePermission(
                    uid=uid,
                    server_uid=self.server_uid,
                )
            )
        return obj

    @as_result(SyftException)
    def take_ownership(self, uid: UID, credentials: SyftVerifyKey) -> bool:
        if uid in self.permissions or uid in self.data:
            raise SyftException(public_message=f"UID: {uid} already owned.")

        # The first person using this UID can claim ownership
        self.add_permissions(
            [
                ActionObjectOWNER(uid=uid, credentials=credentials),
                ActionObjectWRITE(uid=uid, credentials=credentials),
                ActionObjectREAD(uid=uid, credentials=credentials),
                ActionObjectEXECUTE(uid=uid, credentials=credentials),
            ]
        )

        return True

    def add_permission(self, permission: ActionObjectPermission) -> None:
        permissions = self.permissions[permission.uid]
        permissions.add(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        permissions = self.permissions[permission.uid]
        permissions.remove(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        for permission in permissions:
            self.add_permission(permission)

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        if not isinstance(permission.permission, ActionPermission):
            raise Exception(f"ObjectPermission type: {permission.permission} not valid")

        if (
            permission.credentials
            and self.root_verify_key.verify == permission.credentials.verify
        ):
            return True

        if (
            permission.credentials
            and self.has_admin_permissions is not None
            and self.has_admin_permissions(permission.credentials)
        ):
            return True

        if (
            permission.uid in self.permissions
            and permission.permission_string in self.permissions[permission.uid]
        ):
            return True

        # 🟡 TODO 14: add ALL_READ, ALL_EXECUTE etc
        # third party
        if permission.permission == ActionPermission.OWNER:
            pass
        elif (
            permission.permission == ActionPermission.READ
            and ActionObjectPermission(
                permission.uid, ActionPermission.ALL_READ
            ).permission_string
            in self.permissions[permission.uid]
        ):
            return True
        elif permission.permission == ActionPermission.WRITE:
            pass
        elif permission.permission == ActionPermission.EXECUTE:
            pass

        return False

    @as_result(SyftException)
    def _get_permissions_for_uid(self, uid: UID) -> set[str]:
        if uid in self.permissions:
            return self.permissions[uid]
        raise SyftException(public_message=f"No permissions found for uid: {uid}")

    @as_result(SyftException)
    def get_all_permissions(self) -> dict[UID, set[str]]:
        return self.permissions

    def add_storage_permission(self, permission: StoragePermission) -> None:
        permissions = self.storage_permissions[permission.uid]
        permissions.add(permission.server_uid)
        self.storage_permissions[permission.uid] = permissions

    def add_storage_permissions(self, permissions: list[StoragePermission]) -> None:
        for permission in permissions:
            self.add_storage_permission(permission)

    def remove_storage_permission(self, permission: StoragePermission) -> None:
        permissions = self.storage_permissions[permission.uid]
        permissions.remove(permission.server_uid)
        self.storage_permissions[permission.uid] = permissions

    def has_storage_permission(self, permission: StoragePermission | UID) -> bool:
        if isinstance(permission, UID):
            permission = StoragePermission(uid=permission, server_uid=self.server_uid)

        if permission.uid in self.storage_permissions:
            return permission.server_uid in self.storage_permissions[permission.uid]
        return False

    @as_result(SyftException)
    def _get_storage_permissions_for_uid(self, uid: UID) -> set[UID]:
        if uid in self.storage_permissions:
            return self.storage_permissions[uid]
        raise SyftException(
            public_message=f"No storage permissions found for uid: {uid}"
        )

    @as_result(SyftException)
    def get_all_storage_permissions(self) -> dict[UID, set[UID]]:
        return self.storage_permissions

    @as_result(SyftException)
    def _all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool | None = False,
    ) -> list[NewBaseStash.object_type]:  # type: ignore
        # this checks permissions
        res = [self._get(uid, credentials, has_permission) for uid in self.data.keys()]
        result = [x.ok() for x in res if x.is_ok()]
        if order_by is not None:
            result = sorted(result, key=lambda x: getattr(x, order_by.key, ""))
        return result

    @as_result(SyftException)
    def _remove_keys(
        self,
        store_key: QueryKey,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
    ) -> None:
        uqks = unique_query_keys.all
        for qk in uqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.unique_keys[pk_key]
            ck_col.pop(store_key.value, None)
            self.unique_keys[pk_key] = ck_col

        sqks = searchable_query_keys.all
        for qk in sqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.searchable_keys[pk_key]
            if isinstance(pk_value, list):
                for pk_value_item in pk_value:
                    pk_value_str = str(pk_value_item)
                    if pk_value_str in ck_col and (
                        store_key.value in ck_col[pk_value_str]
                    ):
                        ck_col[pk_value_str].remove(store_key.value)
            else:
                if pk_value in ck_col and (store_key.value in ck_col[pk_value]):
                    ck_col[pk_value].remove(store_key.value)
            self.searchable_keys[pk_key] = ck_col

    @as_result(SyftException)
    def _find_index_or_search_keys(
        self,
        credentials: SyftVerifyKey,
        index_qks: QueryKeys,
        search_qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[SyftObject]:
        ids: set | None = None
        errors = []
        # third party
        if len(index_qks.all) > 0:
            index_results = self._get_keys_index(qks=index_qks)
            if index_results.is_ok():
                if ids is None:
                    ids = index_results.ok() if index_results.ok() else set()
                ids = cast(set, ids)
                ids = ids.intersection(index_results.ok())
            else:
                errors.append(index_results.err())

        search_results = None
        if len(search_qks.all) > 0:
            search_results = self._find_keys_search(qks=search_qks)

            if search_results.is_ok():
                if ids is None:
                    ids = search_results.ok() if search_results.ok() else set()
                ids = cast(set, ids)
                ids = ids.intersection(search_results.ok())
            else:
                errors.append(search_results.err())

        if len(errors) > 0:
            raise SyftException(public_message=" ".join([str(e) for e in errors]))

        if ids is None:
            return []

        qks: QueryKeys = self.store_query_keys(ids)
        return self._get_all_from_store(
            credentials=credentials, qks=qks, order_by=order_by
        ).unwrap()

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
        try:
            if qk.value not in self.data:
                raise SyftException(
                    public_message=f"No {type(obj)} exists for query key: {qk}"
                )

            if has_permission or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            ):
                _original_obj = self.data[qk.value]
                _original_unique_keys = self.settings.unique_keys.with_obj(
                    _original_obj
                )
                if allow_missing_keys:
                    searchable_keys = PartitionKeys(
                        pks=[
                            x
                            for x in self.settings.searchable_keys.all
                            if hasattr(_original_obj, x.key)
                        ]
                    )
                    _original_searchable_keys = searchable_keys.with_obj(_original_obj)

                else:
                    _original_searchable_keys = self.settings.searchable_keys.with_obj(
                        _original_obj
                    )

                store_query_key = self.settings.store_key.with_obj(_original_obj)

                # remove old keys
                self._remove_keys(
                    store_key=store_query_key,
                    unique_query_keys=_original_unique_keys,
                    searchable_query_keys=_original_searchable_keys,
                )

                # update the object with new data
                if overwrite:
                    # Overwrite existing object and their values
                    _original_obj = obj
                else:
                    for key, value in obj.to_dict(exclude_empty=True).items():
                        if key == "id":
                            # protected field
                            continue
                        setattr(_original_obj, key, value)

                # update data and keys
                self._set_data_and_keys(
                    store_query_key=store_query_key,
                    unique_query_keys=self.settings.unique_keys.with_obj(_original_obj),
                    searchable_query_keys=self.settings.searchable_keys.with_obj(
                        _original_obj
                    ),
                    # has been updated
                    obj=_original_obj,
                )

                # 🟡 TODO 28: Add locking in this transaction

                return _original_obj
            else:
                raise SyftException(
                    public_message=f"Failed to update obj {obj}, you have no permission"
                )

        except Exception as e:
            raise SyftException.from_exception(e)

    @as_result(SyftException)
    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[SyftObject]:
        matches = []
        for qk in qks.all:
            if qk.value in self.data:
                if self.has_permission(
                    ActionObjectREAD(uid=qk.value, credentials=credentials)
                ):
                    matches.append(self.data[qk.value])
        if order_by is not None:
            matches = sorted(matches, key=lambda x: getattr(x, order_by.key, ""))
        return matches

    def create(self, obj: SyftObject) -> None:
        pass

    @as_result(SyftException)
    def _delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> SyftSuccess:
        try:
            if has_permission or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            ):
                _obj = self.data.pop(qk.value)
                self.permissions.pop(qk.value)
                self.storage_permissions.pop(qk.value)
                self._delete_unique_keys_for(_obj)
                self._delete_search_keys_for(_obj)
                return SyftSuccess(message="Deleted")
            else:
                raise SyftException(
                    public_message=f"Failed to delete with query key {qk}, you have no permission"
                )
        except Exception as e:
            raise SyftException(
                public_message=f"Failed to delete with query key {qk} with error: {e}"
            )

    @as_result(SyftException)
    def _delete_unique_keys_for(self, obj: SyftObject) -> SyftSuccess:
        for _unique_ck in self.unique_cks:
            qk = _unique_ck.with_obj(obj)
            unique_keys = self.unique_keys[qk.key]
            unique_keys.pop(qk.value, None)
            self.unique_keys[qk.key] = unique_keys
        return SyftSuccess(message="Deleted")

    @as_result(SyftException)
    def _delete_search_keys_for(self, obj: SyftObject) -> SyftSuccess:
        for _search_ck in self.searchable_cks:
            qk: QueryKey = _search_ck.with_obj(obj)
            search_keys: defaultdict = self.searchable_keys[qk.key]
            if isinstance(qk.value, list):
                for qk_value in qk.value:
                    search_keys.pop(qk_value, None)
            else:
                search_keys.pop(qk.value, None)
            self.searchable_keys[qk.key] = search_keys
        return SyftSuccess(message="Deleted")

    @as_result(SyftException)
    def _get_keys_index(self, qks: QueryKeys) -> set[Any]:
        try:
            # match AND
            subsets: list = []
            for qk in qks.all:
                subset: set = set()
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.unique_keys:
                    raise SyftException(
                        public_message=f"Failed to query index with {qk}"
                    )
                ck_col = self.unique_keys[pk_key]
                if pk_value not in ck_col.keys():
                    # must be at least one in all query keys
                    continue
                store_value = ck_col[pk_value]
                subsets.append({store_value})

            if len(subsets) == 0:
                return set()
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)

            return subset
        except Exception as e:
            raise SyftException(public_message=f"Failed to query with {qks}. {e}")

    @as_result(SyftException)
    def _find_keys_search(self, qks: QueryKeys) -> set[QueryKey]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset: set = set()
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.searchable_keys:
                    raise SyftException(public_message=f"Failed to search with {qk}")
                ck_col = self.searchable_keys[pk_key]
                if qk.type_list:
                    # 🟡 TODO: change this hacky way to do on to many relationships
                    # this is when you search a QueryKey which is a list of items
                    # at the moment its mostly just a List[UID]
                    # match OR against all keys for this col
                    # the values of the list will be turned into strings in a single key
                    matches = set()
                    for item in pk_value:
                        for col_key in ck_col.keys():
                            if str(item) in col_key:
                                store_values = ck_col[col_key]
                                for value in store_values:
                                    matches.add(value)
                    if len(matches):
                        subsets.append(matches)
                else:
                    # this is the normal path
                    if pk_value not in ck_col.keys():
                        # must be at least one in all query keys
                        subsets.append(set())
                        continue
                    store_values = ck_col[pk_value]
                    subsets.append(set(store_values))

            if len(subsets) == 0:
                return set()
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)
            return subset
        except Exception as e:
            raise SyftException(public_message=f"Failed to query with {qks}. {e}")

    @as_result(SyftException)
    def _check_partition_keys_unique(
        self, unique_query_keys: QueryKeys
    ) -> UniqueKeyCheck:
        # dont check the store key
        qks = [
            x
            for x in unique_query_keys.all
            if x.partition_key != self.settings.store_key
        ]
        matches = []
        for qk in qks:
            pk_key, pk_value = qk.key, qk.value
            if pk_key not in self.unique_keys:
                raise SyftException(
                    public_message=f"pk_key: {pk_key} not in unique_keys: {self.unique_keys.keys()}"
                )
            ck_col = self.unique_keys[pk_key]
            if pk_value in ck_col:
                matches.append(pk_key)

        if len(matches) == 0:
            return UniqueKeyCheck.EMPTY
        elif len(matches) == len(qks):
            return UniqueKeyCheck.MATCHES

        return UniqueKeyCheck.ERROR

    def _set_data_and_keys(
        self,
        store_query_key: QueryKey,
        unique_query_keys: QueryKeys,
        searchable_query_keys: QueryKeys,
        obj: SyftObject,
    ) -> None:
        uqks = unique_query_keys.all

        for qk in uqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.unique_keys[pk_key]
            ck_col[pk_value] = store_query_key.value
            self.unique_keys[pk_key] = ck_col

        self.unique_keys[store_query_key.key][store_query_key.value] = (
            store_query_key.value
        )

        sqks = searchable_query_keys.all
        for qk in sqks:
            pk_key, pk_value = qk.key, qk.value
            ck_col = self.searchable_keys[pk_key]
            if qk.type_list:
                # coerce the list of objects to strings for a single key
                pk_value = " ".join([str(obj) for obj in pk_value])

            # check if key is present, then add to existing key
            if pk_value in ck_col:
                ck_col[pk_value].append(store_query_key.value)
            else:
                # else create the key with a list
                ck_col[pk_value] = [store_query_key.value]

            self.searchable_keys[pk_key] = ck_col

        self.data[store_query_key.value] = obj

    @as_result(SyftException)
    def _migrate_data(
        self, to_klass: SyftObject, context: AuthedServiceContext, has_permission: bool
    ) -> bool:
        credentials = context.credentials
        has_permission = (credentials == self.root_verify_key) or has_permission
        if has_permission:
            for key, value in self.data.items():
                try:
                    migrated_value = value.migrate_to(to_klass.__version__, context)
                except Exception:
                    raise SyftException(
                        public_message=f"Failed to migrate data to {to_klass} for qk {to_klass.__version__}: {key}"
                    )
                qk = self.settings.store_key.with_obj(key)
                self._update(
                    credentials,
                    qk=qk,
                    obj=migrated_value,
                    has_permission=has_permission,
                    overwrite=True,
                    allow_missing_keys=True,
                ).unwrap()

            return True

        raise SyftException(
            public_message="You don't have permissions to migrate data."
        )
