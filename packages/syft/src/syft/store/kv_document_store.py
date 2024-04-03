# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import Any

# third party
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.serializable import serializable
from ..service.action.action_permissions import ActionObjectEXECUTE
from ..service.action.action_permissions import ActionObjectOWNER
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import ActionObjectREAD
from ..service.action.action_permissions import ActionObjectWRITE
from ..service.action.action_permissions import ActionPermission
from ..service.action.action_permissions import StoragePermission
from ..service.context import AuthedServiceContext
from ..service.response import SyftSuccess
from ..types.syft_object import SyftObject
from ..types.uid import UID
from .document_store import BaseStash
from .document_store import PartitionKey
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StorePartition


@serializable()
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

    def init_store(self) -> Result[Ok, Err]:
        store_status = super().init_store()
        if store_status.is_err():
            return store_status

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

            # uid -> set['<node_uid>']
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
            return Err(str(e))

        return Ok(True)

    def __len__(self) -> int:
        return len(self.data)

    def _get(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        has_permission: bool | None = False,
    ) -> Result[SyftObject, str]:
        # relative
        from ..service.action.action_store import ActionObjectREAD

        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)

        if self.has_permission(read_permission) or has_permission:
            syft_object = self.data[uid]
            return Ok(syft_object)
        return Err(f"Permission: {read_permission} denied")

    # Potentially thread-unsafe methods.
    # CAUTION:
    #       * Don't use self.lock here.
    #       * Do not call the public thread-safe methods here(with locking).
    # These methods are called from the public thread-safe API, and will hang the process.

    def _set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        try:
            # if obj.id is None:
            # obj.id = UID()
            store_query_key: QueryKey = self.settings.store_key.with_obj(obj)
            uid = store_query_key.value
            write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
            can_write = self.has_permission(write_permission)
            unique_query_keys: QueryKeys = self.settings.unique_keys.with_obj(obj)
            store_key_exists = store_query_key.value in self.data
            searchable_query_keys = self.settings.searchable_keys.with_obj(obj)

            ck_check = self._check_partition_keys_unique(
                unique_query_keys=unique_query_keys
            )

            if not store_key_exists and ck_check == UniqueKeyCheck.EMPTY:
                # attempt to claim it for writing
                ownership_result = self.take_ownership(uid=uid, credentials=credentials)
                can_write = ownership_result.is_ok()
            elif not ignore_duplicates:
                keys = ", ".join(f"`{key.key}`" for key in unique_query_keys.all)
                return Err(
                    f"Duplication Key Error for {obj}.\n"
                    f"The fields that should be unique are {keys}."
                )
            else:
                # we are not throwing an error, because we are ignoring duplicates
                # we are also not writing though
                return Ok(obj)

            if can_write:
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
                            node_uid=self.node_uid,
                        )
                    )

                return Ok(obj)
            else:
                return Err(f"Permission: {write_permission} denied")
        except Exception as e:
            return Err(f"Failed to write obj {obj}. {e}")

    def take_ownership(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> Result[SyftSuccess, str]:
        # first person using this UID can claim ownership
        if uid not in self.permissions and uid not in self.data:
            self.add_permissions(
                [
                    ActionObjectOWNER(uid=uid, credentials=credentials),
                    ActionObjectWRITE(uid=uid, credentials=credentials),
                    ActionObjectREAD(uid=uid, credentials=credentials),
                    ActionObjectEXECUTE(uid=uid, credentials=credentials),
                ]
            )
            return Ok(SyftSuccess(message=f"Ownership of ID: {uid} taken."))
        return Err(f"UID: {uid} already owned.")

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

        # TODO: fix for other admins
        if (
            permission.credentials
            and self.root_verify_key.verify == permission.credentials.verify
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

    def add_storage_permission(self, permission: StoragePermission) -> None:
        permissions = self.storage_permissions[permission.uid]
        permissions.add(permission.node_uid)
        self.storage_permissions[permission.uid] = permissions

    def add_storage_permissions(self, permissions: list[StoragePermission]) -> None:
        for permission in permissions:
            self.add_storage_permission(permission)

    def remove_storage_permission(self, permission: StoragePermission) -> None:
        permissions = self.storage_permissions[permission.uid]
        permissions.remove(permission.node_uid)
        self.storage_permissions[permission.uid] = permissions

    def has_storage_permission(self, permission: StoragePermission | UID) -> bool:
        if isinstance(permission, UID):
            permission = StoragePermission(uid=permission, node_uid=self.node_uid)

        if permission.uid in self.storage_permissions:
            return permission.node_uid in self.storage_permissions[permission.uid]
        return False

    def _all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool | None = False,
    ) -> Result[list[BaseStash.object_type], str]:
        # this checks permissions
        res = [self._get(uid, credentials, has_permission) for uid in self.data.keys()]
        result = [x.ok() for x in res if x.is_ok()]
        if order_by is not None:
            result = sorted(result, key=lambda x: getattr(x, order_by.key, ""))
        return Ok(result)

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
            if pk_value in ck_col and (store_key.value in ck_col[pk_value]):
                ck_col[pk_value].remove(store_key.value)
            self.searchable_keys[pk_key] = ck_col

    def _find_index_or_search_keys(
        self,
        credentials: SyftVerifyKey,
        index_qks: QueryKeys,
        search_qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> Result[list[SyftObject], str]:
        ids: set | None = None
        errors = []
        # third party
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
            search_results = self._find_keys_search(qks=search_qks)

            if search_results.is_ok():
                if ids is None:
                    ids = search_results.ok()
                ids = ids.intersection(search_results.ok())
            else:
                errors.append(search_results.err())

        if len(errors) > 0:
            return Err(" ".join(errors))

        if ids is None:
            return Ok([])

        qks: QueryKeys = self.store_query_keys(ids)
        return self._get_all_from_store(
            credentials=credentials, qks=qks, order_by=order_by
        )

    def _update(
        self,
        credentials: SyftVerifyKey,
        qk: QueryKey,
        obj: SyftObject,
        has_permission: bool = False,
        overwrite: bool = False,
    ) -> Result[SyftObject, str]:
        try:
            if qk.value not in self.data:
                return Err(f"No object exists for query key: {qk}")

            if has_permission or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            ):
                _original_obj = self.data[qk.value]
                _original_unique_keys = self.settings.unique_keys.with_obj(
                    _original_obj
                )
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

                return Ok(_original_obj)
            else:
                return Err(f"Failed to update obj {obj}, you have no permission")

        except Exception as e:
            return Err(f"Failed to update obj {obj} with error: {e}")

    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> Result[list[SyftObject], str]:
        matches = []
        for qk in qks.all:
            if qk.value in self.data:
                if self.has_permission(
                    ActionObjectREAD(uid=qk.value, credentials=credentials)
                ):
                    matches.append(self.data[qk.value])
        if order_by is not None:
            matches = sorted(matches, key=lambda x: getattr(x, order_by.key, ""))
        return Ok(matches)

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def _delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> Result[SyftSuccess, Err]:
        try:
            if has_permission or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            ):
                _obj = self.data.pop(qk.value)
                self.permissions.pop(qk.value)
                self.storage_permissions.pop(qk.value)
                self._delete_unique_keys_for(_obj)
                self._delete_search_keys_for(_obj)
                return Ok(SyftSuccess(message="Deleted"))
            else:
                return Err(
                    f"Failed to delete with query key {qk}, you have no permission"
                )
        except Exception as e:
            return Err(f"Failed to delete with query key {qk} with error: {e}")

    def _delete_unique_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _unique_ck in self.unique_cks:
            qk = _unique_ck.with_obj(obj)
            unique_keys = self.unique_keys[qk.key]
            unique_keys.pop(qk.value, None)
            self.unique_keys[qk.key] = unique_keys
        return Ok(SyftSuccess(message="Deleted"))

    def _delete_search_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _search_ck in self.searchable_cks:
            qk = _search_ck.with_obj(obj)
            search_keys = self.searchable_keys[qk.key]
            search_keys.pop(qk.value, None)
            self.searchable_keys[qk.key] = search_keys
        return Ok(SyftSuccess(message="Deleted"))

    def _get_keys_index(self, qks: QueryKeys) -> Result[set[Any], str]:
        try:
            # match AND
            subsets: list = []
            for qk in qks.all:
                subset: set = set()
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

    def _find_keys_search(self, qks: QueryKeys) -> Result[set[QueryKey], str]:
        try:
            # match AND
            subsets = []
            for qk in qks.all:
                subset: set = set()
                pk_key, pk_value = qk.key, qk.value
                if pk_key not in self.searchable_keys:
                    return Err(f"Failed to search with {qk}")
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
                return Ok(set())
            # AND
            subset = subsets.pop()
            for s in subsets:
                subset = subset.intersection(s)
            return Ok(subset)
        except Exception as e:
            return Err(f"Failed to query with {qks}. {e}")

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
                raise Exception(
                    f"pk_key: {pk_key} not in unique_keys: {self.unique_keys.keys()}"
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

    def _migrate_data(
        self, to_klass: SyftObject, context: AuthedServiceContext, has_permission: bool
    ) -> Result[bool, str]:
        credentials = context.credentials
        has_permission = (credentials == self.root_verify_key) or has_permission
        if has_permission:
            for key, value in self.data.items():
                try:
                    migrated_value = value.migrate_to(to_klass.__version__, context)
                except Exception:
                    return Err(f"Failed to migrate data to {to_klass} for qk: {key}")
                qk = self.settings.store_key.with_obj(key)
                result = self._update(
                    credentials,
                    qk=qk,
                    obj=migrated_value,
                    has_permission=has_permission,
                    overwrite=True,
                )

                if result.is_err():
                    return result.err()

            return Ok(True)

        return Err("You don't have permissions to migrate data.")
