# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Type
from typing import Union

# third party
from pymongo import ASCENDING
from pymongo.collection import Collection as MongoCollection
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..service.action.action_permissions import ActionObjectEXECUTE
from ..service.action.action_permissions import ActionObjectOWNER
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import ActionObjectREAD
from ..service.action.action_permissions import ActionObjectWRITE
from ..service.action.action_permissions import ActionPermission
from ..service.context import AuthedServiceContext
from ..service.response import SyftSuccess
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
from .kv_document_store import KeyValueBackingStore
from .locks import LockingConfig
from .locks import NoLockingConfig
from .mongo_client import MongoClient
from .mongo_client import MongoStoreClientConfig


@serializable()
class MongoDict(SyftBaseObject):
    __canonical_name__ = "MongoDict"
    __version__ = SYFT_OBJECT_VERSION_1

    keys: List[Any]
    values: List[Any]

    @property
    def dict(self) -> Dict[Any, Any]:
        return dict(zip(self.keys, self.values))

    @staticmethod
    def from_dict(input: Dict[Any, Any]) -> Self:
        return MongoDict(keys=list(input.keys()), values=list(input.values()))

    def __repr__(self):
        return self.dict.__repr__()


class MongoBsonObject(StorableObjectType, dict):
    pass


def _repr_debug_(value: Any) -> str:
    if hasattr(value, "_repr_debug_"):
        return value._repr_debug_()
    return repr(value)


def to_mongo(context: TransformContext) -> TransformContext:
    output = {}
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

    if "id" in context.output:
        output["_id"] = context.output["id"]
    output["__canonical_name__"] = context.obj.__canonical_name__
    output["__version__"] = context.obj.__version__
    output["__blob__"] = _serialize(context.obj, to_bytes=True)
    output["__arepr__"] = _repr_debug_(context.obj)  # a comes first in alphabet
    context.output = output
    return context


@transform(SyftObject, MongoBsonObject)
def syft_obj_to_mongo():
    return [to_mongo]


@transform_method(MongoBsonObject, SyftObject)
def from_mongo(
    storage_obj: Dict, context: Optional[TransformContext] = None
) -> SyftObject:
    return _deserialize(storage_obj["__blob__"], from_bytes=True)


@serializable(attrs=["storage_type"])
class MongoStorePartition(StorePartition):
    """Mongo StorePartition

    Parameters:
        `settings`: PartitionSettings
            PySyft specific settings, used for partitioning and indexing.
        `store_config`: MongoStoreConfig
            Mongo specific configuration
    """

    storage_type: StorableObjectType = MongoBsonObject

    def init_store(self) -> Result[Ok, Err]:
        store_status = super().init_store()
        if store_status.is_err():
            return store_status

        client = MongoClient(config=self.store_config.client_config)

        collection_status = client.with_collection(
            collection_settings=self.settings, store_config=self.store_config
        )
        if collection_status.is_err():
            return collection_status

        collection_permissions_status = client.with_collection_permissions(
            collection_settings=self.settings, store_config=self.store_config
        )
        if collection_permissions_status.is_err():
            return collection_permissions_status

        self._collection = collection_status.ok()
        self._permissions = collection_permissions_status.ok()

        return self._create_update_index()

    # Potentially thread-unsafe methods.
    #
    # CAUTION:
    #       * Don't use self.lock here.
    #       * Do not call the public thread-safe methods here(with locking).
    # These methods are called from the public thread-safe API, and will hang the process.

    def _create_update_index(self) -> Result[Ok, Err]:
        """Create or update mongo database indexes"""
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        def check_index_keys(current_keys, new_index_keys):
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
            return Err(str(e))
        index_name = f"{object_name}_index_name"

        current_index_keys = current_indexes.get(index_name, None)

        if current_index_keys is not None:
            keys_same = check_index_keys(current_index_keys["key"], new_index_keys)
            if keys_same:
                return Ok()

            # Drop current index, since incompatible with current object
            try:
                collection.drop_index(index_or_name=index_name)
            except Exception:
                return Err(
                    f"Failed to drop index for object: {object_name} with index keys: {current_index_keys}"
                )

        # If no new indexes, then skip index creation
        if len(new_index_keys) == 0:
            return Ok()

        try:
            collection.create_index(new_index_keys, unique=True, name=index_name)
        except Exception:
            return Err(
                f"Failed to create index for {object_name} with index keys: {new_index_keys}"
            )

        return Ok()

    @property
    def collection(self) -> Result[MongoCollection, Err]:
        if not hasattr(self, "_collection"):
            res = self.init_store()
            if res.is_err():
                return res

        return Ok(self._collection)

    @property
    def permissions(self) -> Result[MongoCollection, Err]:
        if not hasattr(self, "_permissions"):
            res = self.init_store()
            if res.is_err():
                return res

        return Ok(self._permissions)

    def set(self, *args, **kwargs):
        return self._set(*args, **kwargs)

    def _set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        # TODO: Refactor this function since now it's doing both set and
        # update at the same time
        write_permission = ActionObjectWRITE(uid=obj.id, credentials=credentials)
        can_write = self.has_permission(write_permission)

        store_query_key: QueryKey = self.settings.store_key.with_obj(obj)
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        store_key_exists = (
            collection.find_one(store_query_key.as_dict_mongo) is not None
        )
        if (not store_key_exists) and (not self.item_keys_exist(obj, collection)):
            # attempt to claim ownership for writing
            ownership_result = self.take_ownership(uid=obj.id, credentials=credentials)
            can_write: bool = ownership_result.is_ok()
        elif not ignore_duplicates:
            unique_query_keys: QueryKeys = self.settings.unique_keys.with_obj(obj)
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

            return Ok(obj)
        else:
            return Err(f"No permission to write object with id {obj.id}")

    def item_keys_exist(self, obj, collection):
        qks: QueryKeys = self.settings.unique_keys.with_obj(obj)
        query = {"$or": [{k: v} for k, v in qks.as_dict_mongo.items()]}
        res = collection.find_one(query)
        return res is not None

    def _update(
        self,
        credentials: SyftVerifyKey,
        qk: QueryKey,
        obj: SyftObject,
        has_permission: bool = False,
    ) -> Result[SyftObject, str]:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        # TODO: optimize the update. The ID should not be overwritten,
        # but the qk doesn't necessarily have to include the `id` field either.

        prev_obj_status = self._get_all_from_store(credentials, QueryKeys(qks=[qk]))
        if prev_obj_status.is_err():
            return Err(f"No object found with query key: {qk}")

        prev_obj = prev_obj_status.ok()
        if len(prev_obj) == 0:
            return Err(f"Missing values for query key: {qk}")

        prev_obj = prev_obj[0]
        if has_permission or self.has_permission(
            ActionObjectWRITE(uid=prev_obj.id, credentials=credentials)
        ):
            # we don't want to overwrite Mongo's "id_" or Syft's "id" on update
            obj_id = obj["id"]

            # Set ID to the updated object value
            obj.id = prev_obj["id"]

            # Create the Mongo object
            storage_obj = obj.to(self.storage_type)

            # revert the ID
            obj.id = obj_id

            try:
                collection.update_one(
                    filter=qk.as_dict_mongo, update={"$set": storage_obj}
                )
            except Exception as e:
                return Err(f"Failed to update obj: {obj} with qk: {qk}. Error: {e}")

            return Ok(obj)
        else:
            return Err(f"Failed to update obj {obj}, you have no permission")

    def _find_index_or_search_keys(
        self,
        credentials: SyftVerifyKey,
        index_qks: QueryKeys,
        search_qks: QueryKeys,
        order_by: Optional[PartitionKey] = None,
    ) -> Result[List[SyftObject], str]:
        # TODO: pass index as hint to find method
        qks = QueryKeys(qks=(index_qks.all + search_qks.all))
        return self._get_all_from_store(
            credentials=credentials, qks=qks, order_by=order_by
        )

    @property
    def data(self):
        values: List = self._all(credentials=None, has_permission=True).ok()
        return {v.id: v for v in values}

    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: Optional[PartitionKey] = None,
        has_permission: Optional[bool] = False,
    ) -> Result[List[SyftObject], str]:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        if order_by is not None:
            storage_objs = collection.find(filter=qks.as_dict_mongo).sort(order_by.key)
        else:
            _default_key = "_id"
            storage_objs = collection.find(filter=qks.as_dict_mongo).sort(_default_key)
        syft_objs = []
        for storage_obj in storage_objs:
            obj = self.storage_type(storage_obj)
            transform_context = TransformContext(output={}, obj=obj)
            syft_objs.append(obj.to(self.settings.object_type, transform_context))

        # TODO: maybe do this in loop before this
        res = []
        for s in syft_objs:
            if has_permission or self.has_permission(
                ActionObjectREAD(uid=s.id, credentials=credentials)
            ):
                res.append(s)
        return Ok(res)

    def _delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> Result[SyftSuccess, Err]:
        if not (
            has_permission
            or self.has_permission(
                ActionObjectWRITE(uid=qk.value, credentials=credentials)
            )
        ):
            return Err(f"You don't have permission to delete object with qk: {qk}")

        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        collection_permissions_status = self.permissions
        if collection_permissions_status.is_err():
            return collection_permissions_status
        collection_permissions: MongoCollection = collection_permissions_status.ok()

        qks = QueryKeys(qks=qk)
        # delete the object
        result = collection.delete_one(filter=qks.as_dict_mongo)
        # delete the object's permission
        result_permission = collection_permissions.delete_one(filter=qks.as_dict_mongo)
        if result.deleted_count == 1 and result_permission.deleted_count == 1:
            return Ok(SyftSuccess(message="Object and its permission are deleted"))
        elif result.deleted_count == 0:
            return Err(f"Failed to delete object with qk: {qk}")
        else:
            return Err(
                f"Object with qk: {qk} was deleted, but failed to delete its corresponding permission"
            )

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        """Check if the permission is inside the permission collection"""
        collection_permissions_status = self.permissions
        if collection_permissions_status.is_err():
            return False
        collection_permissions: MongoCollection = collection_permissions_status.ok()

        permissions: Optional[Dict] = collection_permissions.find_one(
            {"_id": permission.uid}
        )

        if permissions is None:
            return False

        # TODO: fix for other admins
        if self.root_verify_key.verify == permission.credentials.verify:
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

    def add_permission(self, permission: ActionObjectPermission) -> Result[None, Err]:
        collection_permissions_status = self.permissions
        if collection_permissions_status.is_err():
            return collection_permissions_status
        collection_permissions: MongoCollection = collection_permissions_status.ok()

        # find the permissions for the given permission.uid
        # e.g. permissions = {"_id": "7b88fdef6bff42a8991d294c3d66f757",
        #                      "permissions": set(["permission_str_1", "permission_str_2"]}}
        permissions: Optional[Dict] = collection_permissions.find_one(
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
            permission_strings: Set = permissions["permissions"]
            permission_strings.add(permission.permission_string)
            collection_permissions.update_one(
                {"_id": permission.uid}, {"$set": {"permissions": permission_strings}}
            )

    def add_permissions(self, permissions: List[ActionObjectPermission]) -> None:
        for permission in permissions:
            self.add_permission(permission)

    def remove_permission(
        self, permission: ActionObjectPermission
    ) -> Result[None, Err]:
        collection_permissions_status = self.permissions
        if collection_permissions_status.is_err():
            return collection_permissions_status
        collection_permissions: MongoCollection = collection_permissions_status.ok()
        permissions: Optional[Dict] = collection_permissions.find_one(
            {"_id": permission.uid}
        )
        if permissions is None:
            return Err(f"permission with UID {permission.uid} not found!")
        permissions_strings: Set = permissions["permissions"]
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
            return Err(f"the permission {permission.permission_string} does not exist!")

    def take_ownership(
        self, uid: UID, credentials: SyftVerifyKey
    ) -> Result[SyftSuccess, str]:
        collection_permissions_status = self.permissions
        if collection_permissions_status.is_err():
            return collection_permissions_status
        collection_permissions: MongoCollection = collection_permissions_status.ok()

        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        data: List[UID] = collection.find_one({"_id": uid})
        permissions: List[UID] = collection_permissions.find_one({"_id": uid})

        # first person using this UID can claim ownership
        if permissions is None and data is None:
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

    def _all(
        self,
        credentials: SyftVerifyKey,
        order_by: Optional[PartitionKey] = None,
        has_permission: Optional[bool] = False,
    ):
        qks = QueryKeys(qks=())
        return self._get_all_from_store(
            credentials=credentials,
            qks=qks,
            order_by=order_by,
            has_permission=has_permission,
        )

    def __len__(self):
        collection_status = self.collection
        if collection_status.is_err():
            return 0
        collection: MongoCollection = collection_status.ok()
        return collection.count_documents(filter={})

    def _migrate_data(
        self, to_klass: SyftObject, context: AuthedServiceContext, has_permission: bool
    ) -> Result[bool, str]:
        credentials = context.credentials
        has_permission = (credentials == self.root_verify_key) or has_permission
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

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
                    return Err(f"Failed to migrate data to {to_klass} for qk: {key}")
                qk = self.settings.store_key.with_obj(key)
                result = self._update(
                    credentials,
                    qk=qk,
                    obj=migrated_value,
                    has_permission=has_permission,
                )

                if result.is_err():
                    return result.err()

            return Ok(True)

        return Err("You don't have permissions to migrate data.")


@serializable()
class MongoDocumentStore(DocumentStore):
    """Mongo Document Store

    Parameters:
        `store_config`: MongoStoreConfig
            Mongo specific configuration, including connection configuration, database name, or client class type.
    """

    partition_type = MongoStorePartition


@serializable(attrs=["index_name", "settings", "store_config"])
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
        ddtype: Optional[type] = None,
    ) -> None:
        self.index_name = index_name
        self.settings = settings
        self.store_config = store_config
        self.client: MongoClient
        self.ddtype = ddtype
        self.init_client()

    def init_client(self) -> Union[None, Err]:
        self.client = MongoClient(config=self.store_config.client_config)

        collection_status = self.client.with_collection(
            collection_settings=self.settings,
            store_config=self.store_config,
            collection_name=f"{self.settings.name}_{self.index_name}",
        )
        if collection_status.is_err():
            return collection_status
        self._collection: MongoCollection = collection_status.ok()

    @property
    def collection(self) -> Result[MongoCollection, Err]:
        if not hasattr(self, "_collection"):
            res = self.init_client()
            if res.is_err():
                return res

        return Ok(self._collection)

    def _exist(self, key: UID) -> bool:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        result: Optional[Dict] = collection.find_one({"_id": key})
        if result is not None:
            return True

        return False

    def _set(self, key: UID, value: Any) -> None:
        if self._exist(key):
            self._update(key, value)
        else:
            collection_status = self.collection
            if collection_status.is_err():
                return collection_status
            collection: MongoCollection = collection_status.ok()
            try:
                bson_data = {
                    "_id": key,
                    f"{key}": _serialize(value, to_bytes=True),
                    "_repr_debug_": _repr_debug_(value),
                }
                collection.insert_one(bson_data)
            except Exception as e:
                raise ValueError(f"Cannot insert data. Error message: {e}")

    def _update(self, key: UID, value: Any) -> None:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()
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
            raise RuntimeError(
                f"Failed to update obj: {key} with value: {value}. Error: {e}"
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        self._set(key, value)

    def _get(self, key: UID) -> Any:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()

        result: Optional[Dict] = collection.find_one({"_id": key})
        if result is not None:
            return _deserialize(result[f"{key}"], from_bytes=True)
        else:
            # raise KeyError(f"{key} does not exist")
            # return an empty set which is the same with SQLiteBackingStore
            return set()

    def __getitem__(self, key: Any) -> Self:
        try:
            return self._get(key)
        except KeyError as e:
            raise e

    def _len(self) -> int:
        collection_status = self.collection
        if collection_status.is_err():
            return 0
        collection: MongoCollection = collection_status.ok()
        return collection.count_documents(filter={})

    def __len__(self) -> int:
        return self._len()

    def _delete(self, key: UID) -> None:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()
        result = collection.delete_one({"_id": key})
        if result.deleted_count != 1:
            raise KeyError(f"{key} does not exist")

    def __delitem__(self, key: str):
        self._delete(key)

    def _delete_all(self) -> None:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection: MongoCollection = collection_status.ok()
        collection.delete_many({})

    def clear(self) -> Self:
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

    def update(self, *args: Any, **kwargs: Any) -> Self:
        """
        Inserts the specified items to the dictionary.
        """
        # 🟡 TODO
        raise NotImplementedError

    def __del__(self):
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
                * FileLockingConfig: file based locking, ideal for same-device different-processes/threads stores.
                * RedisLockingConfig: Redis-based locking, ideal for multi-device stores.
            Defaults to NoLockingConfig.
    """

    client_config: MongoStoreClientConfig
    store_type: Type[DocumentStore] = MongoDocumentStore
    db_name: str = "app"
    backing_store = MongoBackingStore
    # TODO: should use a distributed lock, with RedisLockingConfig
    locking_config: LockingConfig = NoLockingConfig()
