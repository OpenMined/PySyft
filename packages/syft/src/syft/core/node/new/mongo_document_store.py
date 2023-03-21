# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from pymongo import ASCENDING
from pymongo import WriteConcern
from pymongo.errors import DuplicateKeyError
from result import Err
from result import Ok
from result import Result

# relative
from .document_store import DocumentStore
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StoreClientConfig
from .document_store import StoreConfig
from .document_store import StorePartition
from .mongo_client import MongoClient
from .response import SyftSuccess
from .serializable import serializable
from .syft_object import StorableObjectType
from .syft_object import SyftObject
from .syft_object import SyftObjectRegistry
from .transforms import TransformContext
from .transforms import transform
from .transforms import transform_method


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

    output["_id"] = context.output["id"]
    output["__canonical_name__"] = context.obj.__canonical_name__
    output["__version__"] = context.obj.__version__
    output["__obj__"] = context.obj.to_dict()
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
    constructor = SyftObjectRegistry.versioned_class(
        name=storage_obj["__canonical_name__"], version=storage_obj["__version__"]
    )
    if constructor is None:
        raise ValueError(
            "Versioned class should not be None for initialization of SyftObject."
        )
    output = storage_obj["__obj__"]
    for attr, funcs in constructor.__serde_overrides__.items():
        if attr in output:
            output[attr] = funcs[1](output[attr])
    return constructor(**output)


@serializable(recursive_serde=True)
class MongoStorePartition(StorePartition):
    __attr_allowlist__ = [
        "storage_type",
        "settings",
        "store_config",
        "unique_cks",
        "searchable_cks",
    ]
    storage_type: StorableObjectType = MongoBsonObject

    def init_store(self):
        super().init_store()
        self._init_collection()

    def _init_collection(self):
        client = MongoClient.from_config(config=self.store_config.client_config)
        self._collection = client.with_collection(
            collection_settings=self.settings, store_config=self.store_config
        )
        self._create_update_index()

    @property
    def collection(self):
        if not hasattr(self, "_collection"):
            self.init_store()

        return self._collection

    def _create_update_index(self):
        """Create or update mongo database indexes"""

        def check_index_keys(current_keys, new_index_keys):
            current_keys.sort()
            new_index_keys.sort()
            return current_keys == new_index_keys

        syft_obj = self.settings.object_type

        unique_attrs = getattr(syft_obj, "__attr_unique__", [])
        object_name = syft_obj.__canonical_name__

        new_index_keys = [(attr, ASCENDING) for attr in unique_attrs]

        current_indexes = self.collection.index_information()
        index_name = f"{object_name}_index_name"

        current_index_keys = current_indexes.get(index_name, None)

        if current_index_keys is not None:
            keys_same = check_index_keys(current_index_keys["key"], new_index_keys)
            if keys_same:
                return

            # Drop current index, since incompatible with current object
            try:
                self.collection.drop_index(index_or_name=index_name)
            except Exception as e:
                print(
                    f"Failed to drop index for object: {object_name} with index keys: {current_index_keys}"
                )
                raise e

        # If no new indexes, then skip index creation
        if len(new_index_keys) == 0:
            return

        try:
            self.collection.create_index(new_index_keys, unique=True, name=index_name)
            print(f"Index Created for {object_name} with indexes: {new_index_keys}")
        except Exception as e:
            print(
                f"Failed to create index for {object_name} with index keys: {new_index_keys}"
            )
            raise e

    def set(
        self,
        obj: SyftObject,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObject, str]:
        storage_obj = obj.to(self.storage_type)

        collection = self.collection

        if ignore_duplicates:
            collection = collection.with_options(write_concern=WriteConcern(w=0))
        try:
            collection.insert_one(storage_obj)
        except DuplicateKeyError as e:
            return Err(f"Duplicate Key Error for {obj}: {e}")

        return Ok(obj)

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        storage_obj = obj.to(self.storage_type)
        try:
            result = self.collection.update_one(
                filter=qk.as_dict_mongo, update={"$set": storage_obj}
            )
        except Exception as e:
            return Err(f"Failed to update obj: {obj} with qk: {qk}. Error: {e}")

        if result.matched_count == 0:
            return Err(f"No object found with query key: {qk}")
        # elif result.modified_count == 0:
        #     return Err(f"Failed to modify obj: {obj} with qk: {qk}")
        return Ok(obj)

    def find_index_or_search_keys(
        self, index_qks: QueryKeys, search_qks: QueryKeys
    ) -> Result[List[SyftObject], str]:
        # TODO: pass index as hint to find method
        qks = QueryKeys(qks=(index_qks.all + search_qks.all))
        return self.get_all_from_store(qks=qks)

    def get_all_from_store(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        storage_objs = self.collection.find(filter=qks.as_dict_mongo)
        syft_objs = []
        for storage_obj in storage_objs:
            obj = self.storage_type(storage_obj)
            transform_context = TransformContext(output={}, obj=obj)
            syft_objs.append(obj.to(self.settings.object_type, transform_context))
        return Ok(syft_objs)

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        qks = QueryKeys(qks=qk)
        result = self.collection.delete_one(filter=qks.as_dict_mongo)

        if result.deleted_count == 1:
            return Ok(SyftSuccess(message="Deleted"))

        return Err(f"Failed to delete object with qk: {qk}")

    def all(self):
        qks = QueryKeys(qks=())
        return self.get_all_from_store(qks=qks)

    def __len__(self):
        return self._collection.count_documents(filter={})


@serializable(recursive_serde=True)
class MongoDocumentStore(DocumentStore):
    partition_type = MongoStorePartition


@serializable(recursive_serde=True)
class MongoStoreConfig(StoreConfig):
    client_config: StoreClientConfig
    store_type: Type[DocumentStore] = MongoDocumentStore
    db_name: str = "app"
