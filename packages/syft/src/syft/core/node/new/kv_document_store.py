# future
from __future__ import annotations

# stdlib
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import Set

# third party
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from .action_permissions import ActionObjectEXECUTE
from .action_permissions import ActionObjectOWNER
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionObjectWRITE
from .action_permissions import ActionPermission
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .document_store import BaseStash
from .document_store import PartitionSettings
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StoreConfig
from .document_store import StorePartition
from .response import SyftSuccess
from .serializable import serializable
from .syft_object import SyftObject
from .twin_object import TwinObject
from .uid import UID
from .user_roles import ServiceRole


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

    def __delitem__(self, key: str):
        raise NotImplementedError

    def clear(self) -> Self:
        raise NotImplementedError

    def copy(self) -> Self:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> Self:
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

    def __init__(
        self,
        root_verify_key: Optional[SyftVerifyKey],
        settings: PartitionSettings,
        store_config: StoreConfig,
    ):
        super().__init__(settings, store_config)
        if root_verify_key is None:
            root_verify_key = SyftSigningKey.generate().verify_key
        self.root_verify_key = root_verify_key

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
            self.permissions = self.store_config.backing_store(
                "permissions", self.settings, self.store_config, ddtype=set
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

        return Ok()

    def get(
        self, uid: UID, credentials: SyftVerifyKey, skip_permission: bool = False
    ) -> Result[SyftObject, str]:
        # relative
        from .action_store import ActionObjectREAD

        # TODO 🟣 Temporarily added skip permission argument for enclave
        # until permissions are fully integrated
        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)
        # if True:
        if skip_permission or self.has_permission(read_permission):
            syft_object = self.data[uid]
            return Ok(syft_object)
        return Err(f"Permission: {read_permission} denied")

    def get_pointer(
        self, uid: UID, credentials: SyftVerifyKey, node_uid: UID
    ) -> Result[SyftObject, str]:
        try:
            # TODO: is this only for actions?
            # 🟡 TODO 34: do we want pointer read permissions?
            if uid in self.data:
                obj = self.data[uid]
                if isinstance(obj, TwinObject):
                    obj = obj.mock
                obj.syft_point_to(node_uid)
                return Ok(obj)
            return Err("Permission denied")
        except Exception as e:
            return Err(str(e))

    def set(
        # self, obj: SyftObject, ignore_duplicates: bool = False
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        try:
            uid = obj.id
            write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
            can_write = self.has_permission(write_permission)
            store_query_key = self.settings.store_key.with_obj(obj)
            exists = store_query_key.value in self.data
            unique_query_keys = self.settings.unique_keys.with_obj(obj)
            searchable_query_keys = self.settings.searchable_keys.with_obj(obj)
            ck_check = self._validate_partition_keys(
                store_query_key=store_query_key, unique_query_keys=unique_query_keys
            )

            if not exists and ck_check == UniqueKeyCheck.EMPTY:
                # attempt to claim it for writing
                ownership_result = self.take_ownership(uid=uid, credentials=credentials)
                can_write = True if ownership_result.is_ok() else False
            elif not ignore_duplicates:
                return Err(f"Duplication Key Error: {obj}")

            if can_write:
                self._set_data_and_keys(
                    store_query_key=store_query_key,
                    unique_query_keys=unique_query_keys,
                    searchable_query_keys=searchable_query_keys,
                    obj=obj,
                )
                self.data[uid] = obj
                if uid not in self.permissions:
                    # create default permissions
                    self.permissions[uid] = set()
                permission = f"{credentials.verify}_READ"
                permissions = self.permissions[uid]
                permissions.add(permission)
                if add_permissions is not None:
                    permissions.update([x.permission_string for x in add_permissions])
                self.permissions[uid] = permissions
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

    def remove_permission(self, permission: ActionObjectPermission):
        permissions = self.permissions[permission.uid]
        permissions.remove(permission.permission_string)
        self.permissions[permission.uid] = permissions

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        results = []
        for permission in permissions:
            results.append(self.add_permission(permission))

    def get_role_for_verify_key(
        self, verify_key: SyftVerifyKey
    ) -> Optional[ServiceRole]:
        # maybe we need a more efficient data structure for this
        matching_users = [x for x in self.data.values() if x.verify_key == verify_key]
        if not len(matching_users):
            return None
        else:
            return matching_users[0].role

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        if not isinstance(permission.permission, ActionPermission):
            raise Exception(f"ObjectPermission type: {permission.permission} not valid")

        if (
            self.root_verify_key.verify == permission.credentials.verify
            or self.get_role_for_verify_key(permission.credentials) == ServiceRole.ADMIN
        ):
            return True

        if (
            permission.uid in self.permissions
            and permission.permission_string in self.permissions[permission.uid]
        ):
            return True

        # 🟡 TODO 14: add ALL_READ, ALL_EXECUTE etc
        # third party
        # import ipdb
        # ipdb.set_trace()
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

    def all(
        self, credentials: SyftVerifyKey
    ) -> Result[List[BaseStash.object_type], str]:
        # this checks permissions
        res = [self.get(uid, credentials) for uid in self.data.keys()]
        return Ok([x.ok() for x in res if x.is_ok()])

    def __len__(self) -> Result[List[BaseStash.object_type], str]:
        return len(self.data)

    def find_index_or_search_keys(
        self, credentials: SyftVerifyKey, index_qks: QueryKeys, search_qks: QueryKeys
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
        return self.get_all_from_store(credentials=credentials, qks=qks)

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

    def update(
        self,
        credentials: SyftVerifyKey,
        qk: QueryKey,
        obj: SyftObject,
        has_permission=False,
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

                # 🟡 TODO 28: Add locking in this transaction

                # remove old keys
                self.remove_keys(
                    unique_query_keys=_original_unique_keys,
                    searchable_query_keys=_original_searchable_keys,
                )

                # update the object with new data
                for key, value in obj.to_dict(exclude_none=True).items():
                    if key == "id":
                        # protected field
                        continue
                    setattr(_original_obj, key, value)

                # update data and keys
                self._set_data_and_keys(
                    store_query_key=qk,
                    unique_query_keys=self.settings.unique_keys.with_obj(_original_obj),
                    searchable_query_keys=self.settings.searchable_keys.with_obj(
                        _original_obj
                    ),
                    obj=_original_obj,
                )

                return Ok(_original_obj)
            else:
                return Err(f"Failed to update obj {obj}, you have no permission")

        except Exception as e:
            return Err(f"Failed to update obj {obj} with error: {e}")

    def get_all_from_store(
        self, credentials: SyftVerifyKey, qks: QueryKeys
    ) -> Result[List[SyftObject], str]:
        matches = []
        for qk in qks.all:
            if qk.value in self.data:
                if self.has_permission(
                    ActionObjectREAD(uid=qk.value, credentials=credentials)
                ):
                    matches.append(self.data[qk.value])
        return Ok(matches)

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        pass

    def delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission=False
    ) -> Result[SyftSuccess, Err]:
        try:
            if has_permission or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            ):
                _obj = self.data.pop(qk.value)
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
            self.unique_keys[qk.key].pop(qk.value, None)
        return Ok(SyftSuccess(message="Deleted"))

    def _delete_search_keys_for(self, obj: SyftObject) -> Result[SyftSuccess, str]:
        for _search_ck in self.searchable_cks:
            qk = _search_ck.with_obj(obj)
            self.searchable_keys[qk.key].pop(qk.value, None)
        return Ok(SyftSuccess(message="Deleted"))

    def _get_keys_index(self, qks: QueryKeys) -> Result[Set[Any], str]:
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

    def _validate_partition_keys(
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
            if pk_value in ck_col or ck_col.get(pk_value) == store_query_key.value:
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
            if qk.type_list:
                # coerce the list of objects to strings for a single key
                pk_value = " ".join([str(obj) for obj in pk_value])

            ck_col[pk_value].append(store_query_key.value)
            self.searchable_keys[pk_key] = ck_col

        self.data[store_query_key.value] = obj
