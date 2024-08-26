# stdlib
from collections.abc import Callable
from typing import Any
from typing import Set  # noqa: UP035

# third party
from pydantic import Field
from pymongo import ASCENDING
from pymongo.collection import Collection as MongoCollection
from typing_extensions import Self

# relative
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
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
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import StorableObjectType
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.transforms import TransformContext
from ..types.transforms import transform
from ..types.transforms import transform_method
from ..types.uid import UID
from .document_store import DocumentStore
from .document_store import PartitionKey
from .document_store import PartitionSettings
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StoreConfig
from .document_store import StorePartition
from .document_store_errors import NotFoundException
from .kv_document_store import KeyValueBackingStore
from .locks import LockingConfig
from .locks import NoLockingConfig
from .mongo_client import MongoClient
from .mongo_client import MongoStoreClientConfig


@serializable()
class MongoDict(SyftBaseObject):
    __canonical_name__ = "MongoDict"
    __version__ = SYFT_OBJECT_VERSION_1

    keys: list[Any]
    values: list[Any]

    @property
    def dict(self) -> dict[Any, Any]:
        return dict(zip(self.keys, self.values))

    @classmethod
    def from_dict(cls, input: dict) -> Self:
        return cls(keys=list(input.keys()), values=list(input.values()))

    def __repr__(self) -> str:
        return self.dict.__repr__()


class MongoBsonObject(StorableObjectType, dict):
    pass


def _repr_debug_(value: Any) -> str:
    if hasattr(value, "_repr_debug_"):
        return value._repr_debug_()
    return repr(value)


def to_mongo(context: TransformContext) -> TransformContext:
    output = {}
    if context.obj:
        unique_keys_dict = context.obj._syft_unique_keys_dict()
        search_keys_dict = context.obj._syft_searchable_keys_dict()
        all_dict = unique_keys_dict
        all_dict.update(search_keys_dict)
        for k in all_dict:
            value = getattr(context.obj, k, "")
            # if the value is a method, store its value
            if callable(value):
                output[k] = value()
            else:
                output[k] = value

        output["__canonical_name__"] = context.obj.__canonical_name__
        output["__version__"] = context.obj.__version__
        output["__blob__"] = _serialize(context.obj, to_bytes=True)
        output["__arepr__"] = _repr_debug_(context.obj)  # a comes first in alphabet

    if context.output and "id" in context.output:
        output["_id"] = context.output["id"]

    context.output = output

    return context


@transform(SyftObject, MongoBsonObject)
def syft_obj_to_mongo() -> list[Callable]:
    return [to_mongo]


@transform_method(MongoBsonObject, SyftObject)
def from_mongo(
    storage_obj: dict, context: TransformContext | None = None
) -> SyftObject:
    return _deserialize(storage_obj["__blob__"], from_bytes=True)


@serializable(attrs=["storage_type"], canonical_name="MongoStorePartition", version=1)
class MongoStorePartition(StorePartition):
    """Mongo StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for partitioning and indexing.
        `store_config`: MongoStoreConfig
            Mongo specific configuration
    """

    storage_type: type[StorableObjectType] = MongoBsonObject

    @as_result(SyftException)
    def init_store(self) -> bool:
        super().init_store().unwrap()
        client = MongoClient(config=self.store_config.client_config)
        self._collection = client.with_collection(
            collection_settings=self.settings, store_config=self.store_config
        ).unwrap()
        self._permissions = client.with_collection_permissions(
            collection_settings=self.settings, store_config=self.store_config
        ).unwrap()
        self._storage_permissions = client.with_collection_storage_permissions(
            collection_settings=self.settings, store_config=self.store_config
        ).unwrap()
        return self._create_update_index().unwrap()

    # Potentially thread-unsafe methods.
    #
    # CAUTION:
    #       * Don't use self.lock here.
    #       * Do not call the public thread-safe methods here(with locking).
    # These methods are called from the public thread-safe API, and will hang the process.

    @as_result(SyftException)
    def _create_update_index(self) -> bool:
        """Create or update mongo database indexes"""
        collection: MongoCollection = self.collection.unwrap()

        def check_index_keys(
            current_keys: list[tuple[str, int]], new_index_keys: list[tuple[str, int]]
        ) -> bool:
            current_keys.sort()
            new_index_keys.sort()
            return current_keys == new_index_keys

        syft_obj = self.settings.object_type

        unique_attrs = getattr(syft_obj, "__attr_unique__", [])
        object_name = syft_obj.__canonical_name__

        new_index_keys = [(attr, ASCENDING) for attr in unique_attrs]

        try:
            current_indexes = collection.index_information()
        except BaseException as e:
            raise SyftException.from_exception(e)
        index_name = f"{object_name}_index_name"

        current_index_keys = current_indexes.get(index_name, None)

        if current_index_keys is not None:
            keys_same = check_index_keys(current_index_keys["key"], new_index_keys)
            if keys_same:
                return True

            # Drop current index, since incompatible with current object
            try:
                collection.drop_index(index_or_name=index_name)
            except Exception:
                raise SyftException(
                    public_message=(
                        f"Failed to drop index for object: {object_name}"
                        f" with index keys: {current_index_keys}"
                    )
                )

        # If no new indexes, then skip index creation
        if len(new_index_keys) == 0:
            return True

        try:
            collection.create_index(new_index_keys, unique=True, name=index_name)
        except Exception:
            raise SyftException(
                public_message=f"Failed to create index for {object_name} with index keys: {new_index_keys}"
            )

        return True

    @property
    @as_result(SyftException)
    def collection(self) -> MongoCollection:
        if not hasattr(self, "_collection"):
            self.init_store().unwrap()
        return self._collection

    @property
    @as_result(SyftException)
    def permissions(self) -> MongoCollection:
        if not hasattr(self, "_permissions"):
            self.init_store().unwrap()
        return self._permissions

    @property
    @as_result(SyftException)
    def storage_permissions(self) -> MongoCollection:
        if not hasattr(self, "_storage_permissions"):
            self.init_store().unwrap()
        return self._storage_permissions

    @as_result(SyftException)
    def set(self, *args: Any, **kwargs: Any) -> SyftObject:
        return self._set(*args, **kwargs).unwrap()

    @as_result(SyftException)
    def _set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> SyftObject:
        # TODO: Refactor this function since now it's doing both set and
        # update at the same time
        write_permission = ActionObjectWRITE(uid=obj.id, credentials=credentials)
        can_write: bool = self.has_permission(write_permission)

        store_query_key: QueryKey = self.settings.store_key.with_obj(obj)
        collection: MongoCollection = self.collection.unwrap()

        store_key_exists = (
            collection.find_one(store_query_key.as_dict_mongo) is not None
        )
        if (not store_key_exists) and (not self.item_keys_exist(obj, collection)):
            # attempt to claim ownership for writing
            can_write = self.take_ownership(
                uid=obj.id, credentials=credentials
            ).unwrap()
        elif not ignore_duplicates:
            unique_query_keys: QueryKeys = self.settings.unique_keys.with_obj(obj)
            keys = ", ".join(f"`{key.key}`" for key in unique_query_keys.all)
            raise SyftException(
                public_message=f"Duplication Key Error for {obj}.\nThe fields that should be unique are {keys}."
            )
        else:
            # we are not throwing an error, because we are ignoring duplicates
            # we are also not writing though
            return obj

        if not can_write:
            raise SyftException(
                public_message=f"No permission to write object with id {obj.id}"
            )

        storage_obj = obj.to(self.storage_type)

        collection.insert_one(storage_obj)

        # adding permissions
        read_permission = ActionObjectPermission(
            uid=obj.id,
            credentials=credentials,
            permission=ActionPermission.READ,
        )
        self.add_permission(read_permission)

        if add_permissions is not None:
            self.add_permissions(add_permissions)

        if add_storage_permission:
            self.add_storage_permission(
                StoragePermission(
                    uid=obj.id,
                    server_uid=self.server_uid,
                )
            )

        return obj

    def item_keys_exist(self, obj: SyftObject, collection: MongoCollection) -> bool:
        qks: QueryKeys = self.settings.unique_keys.with_obj(obj)
        query = {"$or": [{k: v} for k, v in qks.as_dict_mongo.items()]}
        res = collection.find_one(query)
        return res is not None

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
        collection: MongoCollection = self.collection.unwrap()

        # TODO: optimize the update. The ID should not be overwritten,
        # but the qk doesn't necessarily have to include the `id` field either.

        prev_obj = self._get_all_from_store(credentials, QueryKeys(qks=[qk])).unwrap()
        if len(prev_obj) == 0:
            raise SyftException(
                public_message=f"Failed to update missing values for query key: {qk} for type {type(obj)}"
            )

        prev_obj = prev_obj[0]
        if has_permission or self.has_permission(
            ActionObjectWRITE(uid=prev_obj.id, credentials=credentials)
        ):
            for key, value in obj.to_dict(exclude_empty=True).items():
                # we don't want to overwrite Mongo's "id_" or Syft's "id" on update
                if key == "id":
                    # protected field
                    continue

                # Overwrite the value if the key is already present
                setattr(prev_obj, key, value)

            # Create the Mongo object
            storage_obj = prev_obj.to(self.storage_type)

            try:
                collection.update_one(
                    filter=qk.as_dict_mongo, update={"$set": storage_obj}
                )
            except Exception:
                raise SyftException(f"Failed to update obj: {obj} with qk: {qk}")

            return prev_obj
        else:
            raise SyftException(f"Failed to update obj {obj}, you have no permission")

    @as_result(SyftException)
    def _find_index_or_search_keys(
        self,
        credentials: SyftVerifyKey,
        index_qks: QueryKeys,
        search_qks: QueryKeys,
        order_by: PartitionKey | None = None,
    ) -> list[SyftObject]:
        # TODO: pass index as hint to find method
        qks = QueryKeys(qks=(list(index_qks.all) + list(search_qks.all)))
        return self._get_all_from_store(
            credentials=credentials, qks=qks, order_by=order_by
        ).unwrap()

    @property
    def data(self) -> dict:
        values: list = self._all(credentials=None, has_permission=True).unwrap()
        return {v.id: v for v in values}

    @as_result(SyftException)
    def _get(
        self,
        uid: UID,
        credentials: SyftVerifyKey,
        has_permission: bool | None = False,
    ) -> SyftObject:
        qks = QueryKeys.from_dict({"id": uid})
        res = self._get_all_from_store(
            credentials, qks, order_by=None, has_permission=has_permission
        ).unwrap()
        if len(res) == 0:
            raise NotFoundException
        else:
            return res[0]

    @as_result(SyftException)
    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: PartitionKey | None = None,
        has_permission: bool | None = False,
    ) -> list[SyftObject]:
        collection = self.collection.unwrap()

        if order_by is not None:
            storage_objs = collection.find(filter=qks.as_dict_mongo).sort(order_by.key)
        else:
            _default_key = "_id"
            storage_objs = collection.find(filter=qks.as_dict_mongo).sort(_default_key)

        syft_objs = []
        for storage_obj in storage_objs:
            obj = self.storage_type(storage_obj)
            transform_context = TransformContext(output={}, obj=obj)

            syft_obj = obj.to(self.settings.object_type, transform_context)
            if has_permission or self.has_permission(
                ActionObjectREAD(uid=syft_obj.id, credentials=credentials)
            ):
                syft_objs.append(syft_obj)

        return syft_objs

    @as_result(SyftException)
    def _delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> SyftSuccess:
        if not (
            has_permission
            or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            )
        ):
            raise SyftException(
                public_message=f"You don't have permission to delete object with qk: {qk}"
            )

        collection = self.collection.unwrap()
        collection_permissions: MongoCollection = self.permissions.unwrap()

        qks = QueryKeys(qks=qk)
        # delete the object
        result = collection.delete_one(filter=qks.as_dict_mongo)
        # delete the object's permission
        result_permission = collection_permissions.delete_one(filter=qks.as_dict_mongo)
        if result.deleted_count == 1 and result_permission.deleted_count == 1:
            return SyftSuccess(message="Object and its permission are deleted")
        elif result.deleted_count == 0:
            raise SyftException(public_message=f"Failed to delete object with qk: {qk}")
        else:
            raise SyftException(
                public_message=f"Object with qk: {qk} was deleted, but failed to delete its corresponding permission"
            )

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        """Check if the permission is inside the permission collection"""
        collection_permissions_status = self.permissions
        if collection_permissions_status.is_err():
            return False
        collection_permissions: MongoCollection = collection_permissions_status.ok()

        permissions: dict | None = collection_permissions.find_one(
            {"_id": permission.uid}
        )

        if permissions is None:
            return False

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

        if permission.permission_string in permissions["permissions"]:
            return True

        # check ALL_READ permission
        if (
            permission.permission == ActionPermission.READ
            and ActionObjectPermission(
                permission.uid, ActionPermission.ALL_READ
            ).permission_string
            in permissions["permissions"]
        ):
            return True

        return False

    @as_result(SyftException)
    def _get_permissions_for_uid(self, uid: UID) -> Set[str]:  # noqa: UP006
        collection_permissions = self.permissions.unwrap()
        permissions: dict | None = collection_permissions.find_one({"_id": uid})
        if permissions is None:
            raise SyftException(
                public_message=f"Permissions for object with UID {uid} not found!"
            )
        return set(permissions["permissions"])

    @as_result(SyftException)
    def get_all_permissions(self) -> dict[UID, Set[str]]:  # noqa: UP006
        # Returns a dictionary of all permissions {object_uid: {*permissions}}
        collection_permissions: MongoCollection = self.permissions.unwrap()
        permissions = collection_permissions.find({})
        permissions_dict = {}
        for permission in permissions:
            permissions_dict[permission["_id"]] = permission["permissions"]
        return permissions_dict

    def add_permission(self, permission: ActionObjectPermission) -> None:
        collection_permissions = self.permissions.unwrap()

        # find the permissions for the given permission.uid
        # e.g. permissions = {"_id": "7b88fdef6bff42a8991d294c3d66f757",
        #                      "permissions": set(["permission_str_1", "permission_str_2"]}}
        permissions: dict | None = collection_permissions.find_one(
            {"_id": permission.uid}
        )
        if permissions is None:
            # Permission doesn't exist, add a new one
            collection_permissions.insert_one(
                {
                    "_id": permission.uid,
                    "permissions": {permission.permission_string},
                }
            )
        else:
            # update the permissions with the new permission string
            permission_strings: set = permissions["permissions"]
            permission_strings.add(permission.permission_string)
            collection_permissions.update_one(
                {"_id": permission.uid}, {"$set": {"permissions": permission_strings}}
            )

    def add_permissions(self, permissions: list[ActionObjectPermission]) -> None:
        for permission in permissions:
            self.add_permission(permission)

    def remove_permission(self, permission: ActionObjectPermission) -> None:
        collection_permissions = self.permissions.unwrap()
        permissions: dict | None = collection_permissions.find_one(
            {"_id": permission.uid}
        )
        if permissions is None:
            raise SyftException(
                public_message=f"permission with UID {permission.uid} not found!"
            )
        permissions_strings: set = permissions["permissions"]
        if permission.permission_string in permissions_strings:
            permissions_strings.remove(permission.permission_string)
            if len(permissions_strings) > 0:
                collection_permissions.update_one(
                    {"_id": permission.uid},
                    {"$set": {"permissions": permissions_strings}},
                )
            else:
                collection_permissions.delete_one({"_id": permission.uid})
        else:
            raise SyftException(
                public_message=f"the permission {permission.permission_string} does not exist!"
            )

    def add_storage_permission(self, storage_permission: StoragePermission) -> None:
        storage_permissions_collection: MongoCollection = (
            self.storage_permissions.unwrap()
        )
        storage_permissions: dict | None = storage_permissions_collection.find_one(
            {"_id": storage_permission.uid}
        )
        if storage_permissions is None:
            # Permission doesn't exist, add a new one
            storage_permissions_collection.insert_one(
                {
                    "_id": storage_permission.uid,
                    "server_uids": {storage_permission.server_uid},
                }
            )
        else:
            # update the permissions with the new permission string
            server_uids: set = storage_permissions["server_uids"]
            server_uids.add(storage_permission.server_uid)
            storage_permissions_collection.update_one(
                {"_id": storage_permission.uid},
                {"$set": {"server_uids": server_uids}},
            )

    def add_storage_permissions(self, permissions: list[StoragePermission]) -> None:
        for permission in permissions:
            self.add_storage_permission(permission)

    def has_storage_permission(self, permission: StoragePermission) -> bool:  # type: ignore
        """Check if the storage_permission is inside the storage_permission collection"""
        storage_permissions_collection: MongoCollection = (
            self.storage_permissions.unwrap()
        )
        storage_permissions: dict | None = storage_permissions_collection.find_one(
            {"_id": permission.uid}
        )
        if storage_permissions is None or "server_uids" not in storage_permissions:
            return False
        return permission.server_uid in storage_permissions["server_uids"]

    def remove_storage_permission(self, storage_permission: StoragePermission) -> None:
        storage_permissions_collection = self.storage_permissions.unwrap()
        storage_permissions: dict | None = storage_permissions_collection.find_one(
            {"_id": storage_permission.uid}
        )
        if storage_permissions is None:
            raise SyftException(
                public_message=f"storage permission with UID {storage_permission.uid} not found!"
            )
        server_uids: set = storage_permissions["server_uids"]
        if storage_permission.server_uid in server_uids:
            server_uids.remove(storage_permission.server_uid)
            storage_permissions_collection.update_one(
                {"_id": storage_permission.uid},
                {"$set": {"server_uids": server_uids}},
            )
        else:
            raise SyftException(
                public_message=(
                    f"the server_uid {storage_permission.server_uid} does not exist in the storage permission!"
                )
            )

    def _get_storage_permissions_for_uid(self, uid: UID) -> Set[UID]:  # noqa: UP006
        storage_permissions_collection: MongoCollection = (
            self.storage_permissions.unwrap()
        )
        storage_permissions: dict | None = storage_permissions_collection.find_one(
            {"_id": uid}
        )
        if storage_permissions is None:
            raise SyftException(
                public_message=f"Storage permissions for object with UID {uid} not found!"
            )
        return set(storage_permissions["server_uids"])

    @as_result(SyftException)
    def get_all_storage_permissions(
        self,
    ) -> dict[UID, Set[UID]]:  # noqa: UP006
        # Returns a dictionary of all storage permissions {object_uid: {*server_uids}}
        storage_permissions_collection: MongoCollection = (
            self.storage_permissions.unwrap()
        )
        storage_permissions = storage_permissions_collection.find({})
        storage_permissions_dict = {}
        for storage_permission in storage_permissions:
            storage_permissions_dict[storage_permission["_id"]] = storage_permission[
                "server_uids"
            ]
        return storage_permissions_dict

    @as_result(SyftException)
    def take_ownership(self, uid: UID, credentials: SyftVerifyKey) -> bool:
        collection_permissions: MongoCollection = self.permissions.unwrap()
        collection: MongoCollection = self.collection.unwrap()
        data: list[UID] | None = collection.find_one({"_id": uid})
        permissions: list[UID] | None = collection_permissions.find_one({"_id": uid})

        if permissions is not None or data is not None:
            raise SyftException(public_message=f"UID: {uid} already owned.")

        # first person using this UID can claim ownership
        self.add_permissions(
            [
                ActionObjectOWNER(uid=uid, credentials=credentials),
                ActionObjectWRITE(uid=uid, credentials=credentials),
                ActionObjectREAD(uid=uid, credentials=credentials),
                ActionObjectEXECUTE(uid=uid, credentials=credentials),
            ]
        )

        return True

    @as_result(SyftException)
    def _all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool | None = False,
    ) -> list[SyftObject]:
        qks = QueryKeys(qks=())
        return self._get_all_from_store(
            credentials=credentials,
            qks=qks,
            order_by=order_by,
            has_permission=has_permission,
        ).unwrap()

    def __len__(self) -> int:
        collection_status = self.collection
        if collection_status.is_err():
            return 0
        collection: MongoCollection = collection_status.ok()
        return collection.count_documents(filter={})

    @as_result(SyftException)
    def _migrate_data(
        self, to_klass: SyftObject, context: AuthedServiceContext, has_permission: bool
    ) -> bool:
        credentials = context.credentials
        has_permission = (credentials == self.root_verify_key) or has_permission
        collection: MongoCollection = self.collection.unwrap()

        if has_permission:
            storage_objs = collection.find({})
            for storage_obj in storage_objs:
                obj = self.storage_type(storage_obj)
                transform_context = TransformContext(output={}, obj=obj)
                value = obj.to(self.settings.object_type, transform_context)
                key = obj.get("_id")
                try:
                    migrated_value = value.migrate_to(to_klass.__version__, context)
                except Exception:
                    raise SyftException(
                        public_message=f"Failed to migrate data to {to_klass} for qk: {key}"
                    )
                qk = self.settings.store_key.with_obj(key)
                self._update(
                    credentials,
                    qk=qk,
                    obj=migrated_value,
                    has_permission=has_permission,
                ).unwrap()
            return True
        raise SyftException(
            public_message="You don't have permissions to migrate data."
        )


@serializable(canonical_name="MongoDocumentStore", version=1)
class MongoDocumentStore(DocumentStore):
    """Mongo Document Store

    Parameters:
        `store_config`: MongoStoreConfig
            Mongo specific configuration, including connection configuration, database name, or client class type.
    """

    partition_type = MongoStorePartition


@serializable(
    attrs=["index_name", "settings", "store_config"],
    canonical_name="MongoBackingStore",
    version=1,
)
class MongoBackingStore(KeyValueBackingStore):
    """
    Core logic for the MongoDB key-value store

    Parameters:
        `index_name`: str
            Index name (can be either 'data' or 'permissions')
        `settings`: PartitionSettings
            Syft specific settings
        `store_config`: StoreConfig
            Connection Configuration
         `ddtype`: Type
            Optional and should be None
            Used to make a consistent interface with SQLiteBackingStore
    """

    def __init__(
        self,
        index_name: str,
        settings: PartitionSettings,
        store_config: StoreConfig,
        ddtype: type | None = None,
    ) -> None:
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self.client: MongoClient
        self.ddtype = ddtype
        self.init_client()

    @as_result(SyftException)
    def init_client(self) -> None:
        self.client = MongoClient(config=self.store_config.client_config)
        self._collection: MongoCollection = self.client.with_collection(
            collection_settings=self.settings,
            store_config=self.store_config,
            collection_name=f"{self.settings.name}_{self.index_name}",
        ).unwrap()

    @property
    @as_result(SyftException)
    def collection(self) -> MongoCollection:
        if not hasattr(self, "_collection"):
            self.init_client().unwrap()
        return self._collection

    def _exist(self, key: UID) -> bool:
        collection: MongoCollection = self.collection.unwrap()
        result: dict | None = collection.find_one({"_id": key})
        if result is not None:
            return True
        return False

    def _set(self, key: UID, value: Any) -> None:
        if self._exist(key):
            self._update(key, value)
        else:
            collection: MongoCollection = self.collection.unwrap()
            try:
                bson_data = {
                    "_id": key,
                    f"{key}": _serialize(value, to_bytes=True),
                    "_repr_debug_": _repr_debug_(value),
                }
                collection.insert_one(bson_data)
            except Exception:
                raise SyftException(public_message="Cannot insert data.")

    def _update(self, key: UID, value: Any) -> None:
        collection: MongoCollection = self.collection.unwrap()
        try:
            collection.update_one(
                {"_id": key},
                {
                    "$set": {
                        f"{key}": _serialize(value, to_bytes=True),
                        "_repr_debug_": _repr_debug_(value),
                    }
                },
            )
        except Exception as e:
            raise SyftException(
                public_message=f"Failed to update obj: {key} with value: {value}. Error: {e}"
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        self._set(key, value)

    def _get(self, key: UID) -> Any:
        collection: MongoCollection = self.collection.unwrap()
        result: dict | None = collection.find_one({"_id": key})
        if result is not None:
            return _deserialize(result[f"{key}"], from_bytes=True)
        else:
            raise KeyError(f"{key} does not exist")

    def __getitem__(self, key: Any) -> Self:
        try:
            return self._get(key)
        except KeyError as e:
            if self.ddtype is not None:
                return self.ddtype()
            raise e

    def _len(self) -> int:
        collection: MongoCollection = self.collection.unwrap()
        return collection.count_documents(filter={})

    def __len__(self) -> int:
        return self._len()

    def _delete(self, key: UID) -> SyftSuccess:
        collection: MongoCollection = self.collection.unwrap()
        result = collection.delete_one({"_id": key})
        if result.deleted_count != 1:
            raise SyftException(public_message=f"{key} does not exist")
        return SyftSuccess(message="Deleted")

    def __delitem__(self, key: str) -> None:
        self._delete(key)

    def _delete_all(self) -> None:
        collection: MongoCollection = self.collection.unwrap()
        collection.delete_many({})

    def clear(self) -> None:
        self._delete_all()

    def _get_all(self) -> Any:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()
        result = collection.find()
        keys, values = [], []
        for row in result:
            keys.append(row["_id"])
            values.append(_deserialize(row[f"{row['_id']}"], from_bytes=True))
        return dict(zip(keys, values))

    def keys(self) -> Any:
        return self._get_all().keys()

    def values(self) -> Any:
        return self._get_all().values()

    def items(self) -> Any:
        return self._get_all().items()

    def pop(self, key: Any) -> Self:
        value = self._get(key)
        self._delete(key)
        return value

    def __contains__(self, key: Any) -> bool:
        return self._exist(key)

    def __iter__(self) -> Any:
        return iter(self.keys())

    def __repr__(self) -> str:
        return repr(self._get_all())

    def copy(self) -> Self:
        # 🟡 TODO
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Inserts the specified items to the dictionary.
        """
        # 🟡 TODO
        raise NotImplementedError

    def __del__(self) -> None:
        """
        Close the mongo client connection:
            - Cleanup client resources and disconnect from MongoDB
            - End all server sessions created by this client
            - Close all sockets in the connection pools and stop the monitor threads
        """
        self.client.close()


@serializable()
class MongoStoreConfig(StoreConfig):
    __canonical_name__ = "MongoStoreConfig"
    """Mongo Store configuration

    Parameters:
        `client_config`: MongoStoreClientConfig
            Mongo connection details: hostname, port, user, password etc.
        `store_type`: Type[DocumentStore]
            The type of the DocumentStore. Default: MongoDocumentStore
        `db_name`: str
            Database name
        locking_config: LockingConfig
            The config used for store locking. Available options:
                * NoLockingConfig: no locking, ideal for single-thread stores.
                * ThreadingLockingConfig: threading-based locking, ideal for same-process in-memory stores.
            Defaults to NoLockingConfig.
    """

    client_config: MongoStoreClientConfig
    store_type: type[DocumentStore] = MongoDocumentStore
    db_name: str = "app"
    backing_store: type[KeyValueBackingStore] = MongoBackingStore
    # TODO: should use a distributed lock, with RedisLockingConfig
    locking_config: LockingConfig = Field(default_factory=NoLockingConfig)
