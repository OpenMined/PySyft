# stdlib
from typing import List
from typing import Set

# third party
import pymongo
from pymongo.collection import Collection as PyMongoCollection
from result import Err
from result import Ok
from result import Result

# relative
from ..common.node_table.syft_object import SyftObject
from .document_store import BaseCollection
from .document_store import CollectionSettings
from .document_store import DocumentStore
from .document_store import QueryKey
from .document_store import QueryKeys
from .mongo_client import MongoClient
from .mongo_client import MongoClientSettings
from .response import SyftSuccess


class MongoCollectionSettings(CollectionSettings):
    db_name: str


class MongoCollection(BaseCollection):
    db_collection: PyMongoCollection = None

    def __init__(self, settings: CollectionSettings) -> None:
        self.settings = settings
        self.init_store()

    def init_store(self):
        client = MongoClient.from_settings(settings=MongoClientSettings)
        self.db_collection = client.with_collection(collection_settings=self.settings)

    def set(
        self,
        obj: SyftObject,
    ) -> Result[SyftObject, str]:
        bson_dict = obj.to_mongo()
        try:
            self.db_collection.insert_one(bson_dict)
        except pymongo.errors.DuplicateKeyError as e:
            return Err(
                f"Duplicate Key Error: {e}",
            )
        return Ok(obj)

    def update(self, qk: QueryKey, obj: SyftObject) -> Result[SyftObject, str]:
        # 🟡 TODO: 31 Ids in Mongo are like _id, instead of `id`.
        # Maybe we need a method to format query keys according to store type.
        filter_params = {f"_{qk.key}": qk.value.value}
        bson_dict = obj.to_mongo()
        try:
            result = self.db_collection.update_one(
                filter=filter_params, update={"$set": bson_dict}
            )
        except Exception as e:
            return Err(f"Failed to update obj: {obj} with qk: {qk}. Error: {e}")

        if result.modified_count == 0:
            return Err(f"Failed to modify obj: {obj} with qk: {qk}")

        return Ok(obj)

    def get_all_from_store(self, qks: QueryKeys) -> Result[List[SyftObject], str]:
        raise NotImplementedError

    def get_keys_index(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        raise NotImplementedError

    def find_keys_search(self, qks: QueryKeys) -> Result[Set[QueryKey], str]:
        raise NotImplementedError

    def create(self, obj: SyftObject) -> Result[SyftObject, str]:
        raise NotImplementedError

    def delete(self, qk: QueryKey) -> Result[SyftSuccess, Err]:
        raise NotImplementedError


class MongoDocumentStore(DocumentStore):
    collection_type = MongoCollection
