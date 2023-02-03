# stdlib
from typing import Any
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey
from .document_store import BaseStash
from .document_store import CollectionKey
from .document_store import CollectionSettings
from .document_store import DocumentStore
from .document_store import QueryKeys
from .document_store import UIDCollectionKey
from .user import User

# 🟡 TODO 27: it would be nice if these could be defined closer to the User
EmailCollectionKey = CollectionKey(key="email", type_=str)
SigningKeyCollectionKey = CollectionKey(key="signing_key", type_=SyftSigningKey)
VerifyKeyCollectionKey = CollectionKey(key="verify_key", type_=SyftVerifyKey)


@serializable(recursive_serde=True)
class UserStash(BaseStash):
    object_type = User
    settings: CollectionSettings = CollectionSettings(
        name=User.__canonical_name__, object_type=User
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def check_type(self, obj: Any, type_: type) -> Result[Any, str]:
        return (
            Ok(obj)
            if isinstance(obj, type_)
            else Err(f"{type(obj)} does not match required type: {type_}")
        )

    def set(self, user: User) -> Result[User, str]:
        return self.check_type(user, self.object_type).and_then(super().set)

    def get_by_uid(self, uid: UID) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[UIDCollectionKey.with_obj(uid)])
        return Ok(self.query_one(qks=qks))

    def get_by_email(self, email: str) -> Result[Optional[User], str]:
        qks = QueryKeys(qks=[EmailCollectionKey.with_obj(email)])
        return self.query_one(qks=qks)

    def get_by_signing_key(
        self, signing_key: SigningKeyCollectionKey
    ) -> Result[Optional[User], str]:
        if isinstance(signing_key, str):
            signing_key = SyftSigningKey.from_string(signing_key)
        qks = QueryKeys(qks=[SigningKeyCollectionKey.with_obj(signing_key)])
        return self.query_one(qks=qks)

    def get_by_verify_key(
        self, verify_key: VerifyKeyCollectionKey
    ) -> Result[Optional[User], str]:
        if isinstance(verify_key, str):
            verify_key = SyftVerifyKey.from_string(verify_key)
        qks = QueryKeys(qks=[VerifyKeyCollectionKey.with_obj(verify_key)])
        return self.query_one(qks=qks)

    def delete_by_uid(self, uid: UID) -> Result[bool, str]:
        qk = UIDCollectionKey.with_obj(uid)
        return super().delete(qk=qk)

    def update(self, user: User) -> Result[User, str]:
        return self.check_type(user, self.object_type).and_then(super().update)


# def set(self, user: Any) -> None:
#     self.collection.set(syft_object=user)

# def get(self, uid: UID) -> Any:
#     return self.collection.get(uid=uid)

# def delete
# def exists(self, uid: UID) -> Result[bool, str]:

# def __init__(self, client: MongoClient, db_name: str) -> None:
#     self._client = client
#     self._database = client[db_name]
#     self._collection = self._database[self._collection_name]

# def add(self, obj: SyftObject) -> SyftObject:
#     self._collection.insert_one(obj.to_mongo())
#     return obj

# def drop(self) -> None:
#     self._collection.drop()

# def delete(self, **search_params: Any) -> None:
#     search_params = convert_to_mongo_id(search_params)
#     self._collection.delete_many(search_params)

# def update(
#     self, search_params: Dict[str, Any], updated_args: Dict[str, Any]
# ) -> None:
#     search_params = convert_to_mongo_id(search_params)
#     obj = self.first(**search_params)
#     attributes: Dict[str, Any] = {}
#     for k, v in updated_args.items():
#         if k not in obj.__attr_state__:  # type: ignore
#             raise ValueError(f"Cannot set an non existing field:{k} to Syft Object")
#         else:
#             setattr(obj, k, v)
#         if k in obj.__attr_searchable__:  # type: ignore
#             attributes[k] = v
#     attributes["__blob__"] = _serialize(obj, to_bytes=True)  # type: ignore
#     attributes = convert_to_mongo_id(attributes)
#     self.update_one(query=search_params, values=attributes)

# def find(self, search_params: Dict[str, Any]) -> List[SyftObject]:
#     search_params = convert_to_mongo_id(search_params)
#     results = []
#     res = self._collection.find(search_params)
#     for d in res:
#         results.append(SyftObject.from_mongo(d))
#     return results

# def find_one(self, search_params: Dict[str, Any]) -> Optional[SyftObject]:
#     search_params = convert_to_mongo_id(search_params)
#     d = self._collection.find_one(search_params)
#     if d is None:
#         return d
#     return SyftObject.from_mongo(d)

# def first(self, **search_params: Any) -> Optional[SyftObject]:
#     search_params = convert_to_mongo_id(search_params)
#     d = self._collection.find_one(search_params)
#     if d is None:
#         raise ObjectNotFoundError
#     return SyftObject.from_mongo(d)

# def __len__(self) -> int:
#     # empty document filter counts all documents.
#     return self._collection.count_documents(filter={})

# def query(self, **search_params: Any) -> List[SyftObject]:
#     """Query db objects filtering by parameters
#     Args:
#         parameters : List of parameters used to filter.
#     """
#     search_params = convert_to_mongo_id(search_params)
#     cursor = self._collection.find(search_params)
#     objects = [SyftObject.from_mongo(obj) for obj in list(cursor)]
#     return objects

# def update_one(self, query: dict, values: dict) -> None:
#     query = convert_to_mongo_id(query)
#     values = convert_to_mongo_id(values)
#     values = {"$set": values}
#     self._collection.update_one(query, values)

# def all(self) -> List[SyftObject]:
#     return self.find({})

# def clear(self) -> None:
#     self.drop()

# def contain(self, **search_params: Any) -> bool:
#     return bool(self.query(**search_params))
