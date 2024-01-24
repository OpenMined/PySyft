# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from pymongo import ASCENDING
from pymongo import WriteConcern
from pymongo.collection import Collection as MongoCollection
from pymongo.errors import DuplicateKeyError
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ..node.credentials import SyftVerifyKey
from ..serde.deserialize import _deserialize
from ..serde.serializable import serializable
from ..serde.serialize import _serialize
from ..service.action.action_permissions import ActionObjectPermission
from ..service.action.action_permissions import ActionObjectREAD
from ..service.action.action_permissions import ActionObjectWRITE
from ..service.response import SyftSuccess
from ..types.syft_object import StorableObjectType
from ..types.syft_object import SyftBaseObject
from ..types.syft_object import SyftObject
from ..types.transforms import TransformContext
from ..types.transforms import transform
from ..types.transforms import transform_method
from .document_store import DocumentStore
from .document_store import PartitionKey
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StoreConfig
from .document_store import StorePartition
from .locks import LockingConfig
from .locks import NoLockingConfig
from .mongo_client import MongoClient
from .mongo_client import MongoStoreClientConfig


@serializable()
class MongoDict(SyftBaseObject):
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

        self._collection = collection_status.ok()

        return self._create_update_index()

    # Potentially thread-unsafe methods.
    # CAUTION:
    #       * Don't use self.lock here.
    #       * Do not call the public thread-safe methods here(with locking).
    # These methods are called from the public thread-safe API, and will hang the process.

    def _create_update_index(self) -> Result[Ok, Err]:
        """Create or update mongo database indexes"""
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection = collection_status.ok()

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

    def _set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftObject,
        add_permissions: Optional[List[ActionObjectPermission]] = None,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        write_permission = ActionObjectWRITE(uid=obj.id, credentials=credentials)
        can_write = self.has_permission(write_permission)
        if can_write:
            storage_obj = obj.to(self.storage_type)

            collection_status = self.collection
            if collection_status.is_err():
                return collection_status
            collection = collection_status.ok()

            if ignore_duplicates:
                collection = collection.with_options(write_concern=WriteConcern(w=0))
            try:
                collection.insert_one(storage_obj)
            except DuplicateKeyError as e:
                return Err(f"Duplicate Key Error for {obj}: {e}")
            if add_permissions is not None:
                pass
                # TODO: update permissions
            return Ok(obj)
        else:
            return Err(f"No permission to write object with id {obj.id}")

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
        collection = collection_status.ok()

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

    def _get_all_from_store(
        self,
        credentials: SyftVerifyKey,
        qks: QueryKeys,
        order_by: Optional[PartitionKey] = None,
    ) -> Result[List[SyftObject], str]:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection = collection_status.ok()

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
            if self.has_permission(ActionObjectREAD(uid=s.id, credentials=credentials)):
                res.append(s)
        return Ok(res)

    def _delete(
        self, credentials: SyftVerifyKey, qk: QueryKey, has_permission: bool = False
    ) -> Result[SyftSuccess, Err]:
        collection_status = self.collection
        if collection_status.is_err():
            return collection_status
        collection = collection_status.ok()

        if has_permission or self.has_permission(
            ActionObjectWRITE(uid=qk.value, credentials=credentials)
        ):
            qks = QueryKeys(qks=qk)
            result = collection.delete_one(filter=qks.as_dict_mongo)

            if result.deleted_count == 1:
                return Ok(SyftSuccess(message="Deleted"))

        return Err(f"Failed to delete object with qk: {qk}")

    def has_permission(self, permission: ActionObjectPermission) -> bool:
        # TODO: implement
        return True

    def _all(self, credentials: SyftVerifyKey, order_by: Optional[PartitionKey] = None):
        qks = QueryKeys(qks=())
        return self._get_all_from_store(
            credentials=credentials, qks=qks, order_by=order_by
        )

    def __len__(self):
        collection_status = self.collection
        if collection_status.is_err():
            return 0
        collection = collection_status.ok()
        return collection.count_documents(filter={})


@serializable()
class MongoDocumentStore(DocumentStore):
    """Mongo Document Store

    Parameters:
        `store_config`: MongoStoreConfig
            Mongo specific configuration, including connection configuration, database name, or client class type.
    """

    partition_type = MongoStorePartition


@serializable()
class MongoStoreConfig(StoreConfig):
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
    # TODO: should use a distributed lock, with RedisLockingConfig
    locking_config: LockingConfig = NoLockingConfig()
