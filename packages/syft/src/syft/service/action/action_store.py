# future
from __future__ import annotations

# stdlib
import threading

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.dict_document_store import DictStoreConfig
from ...store.document_store import BasePartitionSettings
from ...store.document_store import DocumentStore
from ...store.document_store import StoreConfig
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import ObjectCRUDPermissionException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import SyftObject
from ...types.twin_object import TwinObject
from ...types.uid import LineageID
from ...types.uid import UID
from .action_object import is_action_data_empty
from .action_permissions import ActionObjectEXECUTE
from .action_permissions import ActionObjectOWNER
from .action_permissions import ActionObjectPermission
from .action_permissions import ActionObjectREAD
from .action_permissions import ActionObjectWRITE
from .action_permissions import ActionPermission
from .action_permissions import StoragePermission

lock = threading.RLock()


class ActionStore:
    pass


@serializable(canonical_name="KeyValueActionStore", version=1)
class KeyValueActionStore(ActionStore):
    """Generic Key-Value Action store.

    Parameters:
        store_config: StoreConfig
            Backend specific configuration, including connection configuration, database name, or client class type.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    def __init__(
        self,
        server_uid: UID,
        store_config: StoreConfig,
        root_verify_key: SyftVerifyKey | None = None,
        document_store: DocumentStore | None = None,
    ) -> None:
        self.server_uid = server_uid
        self.store_config = store_config
        self.settings = BasePartitionSettings(name="Action")
        self.data = self.store_config.backing_store(
            "data", self.settings, self.store_config
        )
        self.permissions = self.store_config.backing_store(
            "permissions", self.settings, self.store_config, ddtype=set
        )
        self.storage_permissions = self.store_config.backing_store(
            "storage_permissions", self.settings, self.store_config, ddtype=set
        )

        if root_verify_key is None:
            root_verify_key = SyftSigningKey.generate().verify_key
        self.root_verify_key = root_verify_key

        self.__user_stash = None
        if document_store is not None:
            # relative
            from ...service.user.user_stash import UserStash

            self.__user_stash = UserStash(store=document_store)

    @as_result(NotFoundException, SyftException)
    def get(
        self, uid: UID, credentials: SyftVerifyKey, has_permission: bool = False
    ) -> SyftObject:
        uid = uid.id  # We only need the UID from LineageID or UID

        # if you get something you need READ permission
        read_permission = ActionObjectREAD(uid=uid, credentials=credentials)

        if not has_permission and not self.has_permission(read_permission):
            raise SyftException(public_message=f"Permission: {read_permission} denied")

        # TODO: Remove try/except?
        try:
            if isinstance(uid, LineageID):
                syft_object = self.data[uid.id]
            elif isinstance(uid, UID):
                syft_object = self.data[uid]
            else:
                raise SyftException(
                    public_message=f"Unrecognized UID type: {type(uid)}"
                )
            return syft_object
        except Exception as e:
            raise NotFoundException.from_exception(
                e, public_message=f"Object {uid} not found"
            )

    @as_result(NotFoundException, SyftException)
    def get_mock(self, uid: UID) -> SyftObject:
        uid = uid.id  # We only need the UID from LineageID or UID

        try:
            syft_object = self.data[uid]

            if isinstance(syft_object, TwinObject) and not is_action_data_empty(
                syft_object.mock
            ):
                return syft_object.mock
            raise NotFoundException(public_message=f"No mock found for object {uid}")
        except Exception as e:
            raise NotFoundException.from_exception(
                e, public_message=f"Object {uid} not found"
            )

    @as_result(NotFoundException, SyftException)
    def get_pointer(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        server_uid: UID,
    ) -> SyftObject:
        uid = uid.id  # We only need the UID from LineageID or UID

        try:
            if uid not in self.data:
                raise SyftException(public_message="Permission denied")

            obj = self.data[uid]
            read_permission = ActionObjectREAD(uid=uid, credentials=credentials)

            # if you have permission you can have private data
            if self.has_permission(read_permission):
                if isinstance(obj, TwinObject):
                    return obj.private.syft_point_to(server_uid)
                return obj.syft_point_to(server_uid)

            # if its a twin with a mock anyone can have this
            if isinstance(obj, TwinObject):
                return obj.mock.syft_point_to(server_uid)

            # finally worst case you get ActionDataEmpty so you can still trace
            return obj.as_empty().syft_point_to(server_uid)
        # TODO: Check if this can be removed
        except Exception as e:
            raise SyftException(public_message=str(e))

    def exists(self, uid: UID) -> bool:
        return uid.id in self.data  # We only need the UID from LineageID or UID

    @as_result(SyftException, StashException)
    def set(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        syft_object: SyftObject,
        has_result_read_permission: bool = False,
        add_storage_permission: bool = True,
    ) -> UID:
        uid = uid.id  # We only need the UID from LineageID or UID

        # if you set something you need WRITE permission
        write_permission = ActionObjectWRITE(uid=uid, credentials=credentials)
        can_write = self.has_permission(write_permission)

        if not self.exists(uid=uid):
            # attempt to claim it for writing
            if has_result_read_permission:
                ownership_result = self.take_ownership(uid=uid, credentials=credentials)
                can_write = True if ownership_result.is_ok() else False
            else:
                # root takes owneship, but you can still write
                ownership_result = self.take_ownership(
                    uid=uid, credentials=self.root_verify_key
                )
                can_write = True if ownership_result.is_ok() else False

        if not can_write:
            raise SyftException(public_message=f"Permission: {write_permission} denied")

        self.data[uid] = syft_object
        if uid not in self.permissions:
            # create default permissions
            self.permissions[uid] = set()
        if has_result_read_permission:
            self.add_permission(ActionObjectREAD(uid=uid, credentials=credentials))
        else:
            self.add_permissions(
                [
                    ActionObjectWRITE(uid=uid, credentials=credentials),
                    ActionObjectEXECUTE(uid=uid, credentials=credentials),
                ]
            )

        if uid not in self.storage_permissions:
            # create default storage permissions
            self.storage_permissions[uid] = set()
        if add_storage_permission:
            self.add_storage_permission(
                StoragePermission(uid=uid, server_uid=self.server_uid)
            )

        return uid

    @as_result(SyftException)
    def take_ownership(self, uid: UID, credentials: SyftVerifyKey) -> bool:
        uid = uid.id  # We only need the UID from LineageID or UID

        # first person using this UID can claim ownership
        if uid in self.permissions or uid in self.data:
            raise SyftException(public_message=f"Object {uid} already owned")

        self.add_permissions(
            [
                ActionObjectOWNER(uid=uid, credentials=credentials),
                ActionObjectWRITE(uid=uid, credentials=credentials),
                ActionObjectREAD(uid=uid, credentials=credentials),
                ActionObjectEXECUTE(uid=uid, credentials=credentials),
            ]
        )

        return True

    @as_result(StashException)
    def delete(self, uid: UID, credentials: SyftVerifyKey) -> UID:
        uid = uid.id  # We only need the UID from LineageID or UID

        # if you delete something you need OWNER permission
        # is it bad to evict a key and have someone else reuse it?
        # perhaps we should keep permissions but no data?
        owner_permission = ActionObjectOWNER(uid=uid, credentials=credentials)

        if not self.has_permission(owner_permission):
            raise StashException(
                public_message=f"Permission: {owner_permission} denied"
            )

        if uid in self.data:
            del self.data[uid]
        if uid in self.permissions:
            del self.permissions[uid]

        return uid

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        if not isinstance(permission.permission, ActionPermission):
            # If we reached this point, it's a malformed object error, let it bubble up
            raise TypeError(f"ObjectPermission type: {permission.permission} not valid")

        if (
            permission.credentials is not None
            and self.root_verify_key.verify == permission.credentials.verify
        ):
            return True

        if self.__user_stash is not None:
            # relative
            from ...service.user.user_roles import ServiceRole

            res = self.__user_stash.get_by_verify_key(
                credentials=permission.credentials,
                verify_key=permission.credentials,
            )

            if (
                res.is_ok()
                and (user := res.ok()) is not None
                and user.role in (ServiceRole.DATA_OWNER, ServiceRole.ADMIN)
            ):
                return True

        if (
            permission.uid in self.permissions
            and permission.permission_string in self.permissions[permission.uid]
        ):
            return True

        # 🟡 TODO 14: add ALL_READ, ALL_EXECUTE etc
        if permission.permission == ActionPermission.OWNER:
            pass
        elif permission.permission == ActionPermission.READ:
            pass
        elif permission.permission == ActionPermission.WRITE:
            pass
        elif permission.permission == ActionPermission.EXECUTE:
            pass

        return False

    def has_permissions(self, permissions: list[ActionObjectPermission]) -> bool:
        return all(self.has_permission(p) for p in permissions)

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

    @as_result(ObjectCRUDPermissionException)
    def _get_permissions_for_uid(self, uid: UID) -> set[str]:
        if uid in self.permissions:
            return self.permissions[uid]
        raise ObjectCRUDPermissionException(
            public_message=f"No permissions found for uid: {uid}"
        )

    @as_result(SyftException)
    def get_all_permissions(self) -> dict[UID, set[str]]:
        return dict(self.permissions.items())

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

    @as_result(ObjectCRUDPermissionException)
    def _get_storage_permissions_for_uid(self, uid: UID) -> set[UID]:
        if uid in self.storage_permissions:
            return self.storage_permissions[uid]
        raise ObjectCRUDPermissionException(f"No storage permissions found for {uid}")

    @as_result(SyftException)
    def get_all_storage_permissions(self) -> dict[UID, set[UID]]:
        return dict(self.storage_permissions.items())

    def _all(
        self,
        credentials: SyftVerifyKey,
        has_permission: bool | None = False,
    ) -> list[SyftObject]:
        # this checks permissions
        res = [self.get(uid, credentials, has_permission) for uid in self.data.keys()]
        return [x.ok() for x in res if x.is_ok()]  # type: ignore

    @as_result(ObjectCRUDPermissionException)
    def migrate_data(self, to_klass: SyftObject, credentials: SyftVerifyKey) -> bool:
        has_root_permission = credentials == self.root_verify_key

        if not has_root_permission:
            raise ObjectCRUDPermissionException(
                public_message="You don't have permissions to migrate data."
            )

        for key, value in self.data.items():
            try:
                if value.__canonical_name__ != to_klass.__canonical_name__:
                    continue
                migrated_value = value.migrate_to(to_klass.__version__)
            except Exception as e:
                raise SyftException.from_exception(
                    e,
                    public_message=f"Failed to migrate data to {to_klass} for qk: {key}",
                )

            self.set(
                uid=key,
                credentials=credentials,
                syft_object=migrated_value,
            ).unwrap()

        return True


@serializable(canonical_name="DictActionStore", version=1)
class DictActionStore(KeyValueActionStore):
    """Dictionary-Based Key-Value Action store.

    Parameters:
        store_config: StoreConfig
            Backend specific configuration, including client class type.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    def __init__(
        self,
        server_uid: UID,
        store_config: StoreConfig | None = None,
        root_verify_key: SyftVerifyKey | None = None,
        document_store: DocumentStore | None = None,
    ) -> None:
        store_config = store_config if store_config is not None else DictStoreConfig()
        super().__init__(
            server_uid=server_uid,
            store_config=store_config,
            root_verify_key=root_verify_key,
            document_store=document_store,
        )


@serializable(canonical_name="SQLiteActionStore", version=1)
class SQLiteActionStore(KeyValueActionStore):
    """SQLite-Based Key-Value Action store.

    Parameters:
        store_config: StoreConfig
            SQLite specific configuration, including connection settings or client class type.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    pass


@serializable(canonical_name="MongoActionStore", version=1)
class MongoActionStore(KeyValueActionStore):
    """Mongo-Based  Action store.

    Parameters:
        store_config: StoreConfig
            Mongo specific configuration.
        root_verify_key: Optional[SyftVerifyKey]
            Signature verification key, used for checking access permissions.
    """

    pass
