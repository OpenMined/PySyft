# stdlib
from typing import Dict
from typing import List

# third party
from pymongo.collection import Collection as MongoCollection
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
from ..common.node_table.syft_object import transform
from .credentials import SyftVerifyKey
from .document_store import DocumentStore
from .document_store import QueryKey
from .document_store import QueryKeys
from .document_store import StorePartition
from .mongo_client import MongoClient
from .mongo_client import MongoClientSettings
from .response import SyftSuccess


class MongoBsonObject(StorableObjectType, dict):
    def to_syft_obj(storage_obj: Dict, object_type: SyftObject) -> SyftObject:
        output = deserialize(storage_obj["__blob__"], from_bytes=True)
        for attr, funcs in object_type.__serde_overrides__.items():
            if attr in output:
                output[attr] = funcs[1](output[attr])
        return object_type(**output)


def to_mongo(_self, output) -> Dict:
    output_dict = {}
    for k in _self.__attr_searchable__:
        # 🟡 TODO 24: pass in storage abstraction and detect unsupported types
        # if unsupported, convert to string
        value = getattr(_self, k, "")
        if isinstance(value, SyftVerifyKey):
            value = str(value)
        output_dict[k] = value
    blob = serialize(dict(_self), to_bytes=True)
    output_dict["_id"] = output["id"].value  # type: ignore
    output_dict["__canonical_name__"] = _self.__canonical_name__
    output_dict["__version__"] = _self.__version__
    output_dict["__blob__"] = blob
    output_dict["__repr__"] = _self.__repr__()

    return output_dict


@transform(SyftObject, MongoBsonObject)
def syft_obj_to_mongo():
    return [to_mongo]


class MongoStorePartition(StorePartition):
    db_collection: MongoCollection = None
    storage_type: StorableObjectType = MongoBsonObject

    def init_store(self):
        super().init_store()
        self.init_collection()

    def init_collection(self):
        client = MongoClient.from_settings(settings=MongoClientSettings)
        self.db_collection = client.with_collection(collection_settings=self.settings)

    def set(
        self,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        storage_obj = obj.to(self.storage_type)
        try:
            self.db_collection.insert_one(storage_obj)
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
            result = self.db_collection.update_one(
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
        storage_objs = self.db_collection.find(filter=query_filter)
        syft_objs = [
            self.storage_type.to_syft_obj(
                storage_obj=storage_obj, object_type=self.settings.object_type
            )
            for storage_obj in storage_objs
        ]
        return Ok(syft_objs)

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        query_filter = self._create_filter(qks=QueryKeys(qks=qk))
        result = self.db_collection.delete_one(filter=query_filter)

        if result.deleted_count == 1:
            return Ok(SyftSuccess(message="Deleted"))

        return Err(f"Failed to delete object with qk: {qk}")


@serializable(recursive_serde=True)
class MongoDocumentStore(DocumentStore):
    partition_type = MongoStorePartition
