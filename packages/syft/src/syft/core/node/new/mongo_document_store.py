# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError
from result import Err
from result import Ok
from result import Result

# relative
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import serializable
from ...common.serde.serialize import _serialize as serialize
from ..common.node_table.syft_object import StorableObjectType
from ..common.node_table.syft_object import SyftObject
from ..common.node_table.syft_object import SyftObjectRegistry
from .credentials import SyftVerifyKey
from .document_store import DocumentStore
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StoreConfig
from .document_store import StorePartition
from .mongo_client import MongoClient
from .response import SyftSuccess
from .transforms import TransformContext
from .transforms import transform
from .transforms import transform_method


class MongoBsonObject(StorableObjectType, dict):
    pass


def to_mongo(context: TransformContext) -> TransformContext:
    output_dict = {}
    for k in context.obj.__attr_searchable__:
        # 🟡 TODO 24: pass in storage abstraction and detect unsupported types
        # if unsupported, convert to string
        value = getattr(context.obj, k, "")
        if isinstance(value, SyftVerifyKey):
            value = str(value)
        output_dict[k] = value

    blob = serialize(dict(context.obj), to_bytes=True)
    output_dict["_id"] = context.output["id"].value  # type: ignore
    output_dict["__canonical_name__"] = context.obj.__canonical_name__
    output_dict["__version__"] = context.obj.__version__
    output_dict["__blob__"] = blob
    output_dict["__repr__"] = context.obj.__repr__()
    context.output = output_dict
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
    output = deserialize(storage_obj["__blob__"], from_bytes=True)
    for attr, funcs in constructor.__serde_overrides__.items():
        if attr in output:
            output[attr] = funcs[1](output[attr])
    return constructor(**output)


@serializable(recursive_serde=True)
class MongoStorePartition(StorePartition):
    __attr_allowlist__ = [
        "storage_type",
        "settings",
        "store_client_config",
        "unique_cks",
        "searchable_cks",
    ]
    storage_type: StorableObjectType = MongoBsonObject

    def init_store(self):
        super().init_store()
        self._init_collection()

    def _init_collection(self):
        client = MongoClient.from_config(config=self.store_client_config)
        self._collection = client.with_collection(collection_settings=self.settings)
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
    ) -> Result[SyftObject, str]:
        storage_obj = obj.to(self.storage_type)
        try:
            self.collection.insert_one(storage_obj)
        except DuplicateKeyError as e:
            return Err(f"Duplicate Key Error: {e}")

        return Ok(obj)

    def _create_filter(self, qks: QueryKeys) -> Dict:
        query_filter = {}
        for qk in qks.all:
            qk_key = qk.key
            qk_value = qk.value
            if self.settings.store_key == qk.partition_key:
                qk_key = f"_{qk_key}"
                qk_value = qk_value.value

            query_filter[qk_key] = qk_value

        return query_filter

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        filter_params = self._create_filter(QueryKeys(qks=qk))
        storage_obj = obj.to(self.storage_type)
        try:
            result = self.collection.update_one(
                filter=filter_params, update={"$set": storage_obj}
            )
        except Exception as e:
            return Err(f"Failed to update obj: {obj} with qk: {qk}. Error: {e}")

        if result.matched_count == 0:
            return Err(f"No object found with query key: {qk}")
        elif result.modified_count == 0:
            return Err(f"Failed to modify obj: {obj} with qk: {qk}")
        return Ok(obj)

    def find_index_or_search_keys(
        self, index_qks: QueryKeys, search_qks: QueryKeys
    ) -> Result[List[SyftObject], str]:
        # TODO: pass index as hint to find method
        qks = QueryKeys(qks=(index_qks.all + search_qks.all))
        return self.get_all_from_store(qks=qks)

    def get_all_from_store(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        query_filter = self._create_filter(qks=qks)
        storage_objs = self.collection.find(filter=query_filter)
        syft_objs = []
        for storage_obj in storage_objs:
            obj = self.storage_type(storage_obj)
            transform_context = TransformContext(output={}, obj=obj)
            syft_objs.append(obj.to(self.settings.object_type, transform_context))

        return Ok(syft_objs)

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        query_filter = self._create_filter(qks=QueryKeys(qks=qk))
        result = self.collection.delete_one(filter=query_filter)

        if result.deleted_count == 1:
            return Ok(SyftSuccess(message="Deleted"))

        return Err(f"Failed to delete object with qk: {qk}")

    def all(self):
        qks = QueryKeys(qks=())
        return self.get_all_from_store(qks=qks)


@serializable(recursive_serde=True)
class MongoDocumentStore(DocumentStore):
    partition_type = MongoStorePartition


@serializable(recursive_serde=True)
class MongoStoreConfig(StoreConfig):
    store_type: Type[DocumentStore] = MongoDocumentStore
