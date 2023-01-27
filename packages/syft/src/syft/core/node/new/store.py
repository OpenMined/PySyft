# future
from __future__ import annotations

# stdlib
from typing import Dict
from typing import Optional

# third party
from pymongo.mongo_client import MongoClient

# relative
from ..common.node_table.syft_object import SyftObject


class DocumentStore:
    def __init__(
        self,
        client: Optional[MongoClient] = None,
    ) -> None:
        self.client = client

    def connect_to(
        host_name: str, port: int, username: str, password: str
    ) -> DocumentStore:
        _client = MongoClient(
            host=host_name,
            port=port,
            username=username,
            password=password,
            uuidRepresentation="standard",
        )
        return DocumentStore(client=_client)

    def with_db(self, db_name: str) -> None:
        self._db = self.client[db_name]

    def with_collection(self, collection_name: str, **kwargs) -> None:
        self._collection = self._db.get_collection(name=collection_name, **kwargs)

    def insert(self, obj: SyftObject, in_bulk: bool = False) -> SyftObject:
        if in_bulk:
            # TODO 🟡: to_mongo is able to convert the SyftObject class to list dictionaries
            # Also, we can possibly create a Result object, which is chainable in nature.
            # e.g. store = DocumentStore.connect_to(..)
            # store.with_db("myapp")
            # store.with_collection("User")
            # probably have a custom Result object that is chainable we can possibly do something like this maybe 🤔
            # store.filter({"params"}).sort("id").delete()
            _result = self._collection.insert_many([obj.to_mongo()])
        else:
            _result = self._collection.insert_one(obj.to_mongo())
        return SyftObject.from_mongo(_result)

    def filter(
        self, search_params: Dict, only_one: bool = False, sort_by: Optional[str] = None
    ) -> SyftObject:
        if only_one:
            result = self._collection.find_one(search_params)
        else:
            result = self._collection.find(search_params)
        if sort_by:
            result = result.sort(sort_by)
        return SyftObject.from_mongo(result)
