# stdlib

# stdlib
from typing import cast

# third party
from result import Err
from result import Ok

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.store_errors import StashException
from ...store.store_errors import StashNotFoundException
from ...types.result import catch
from ...types.uid import UID
from ...util.telemetry import instrument
from .user_code import CodeHashPartitionKey
from .user_code import ServiceFuncNamePartitionKey
from .user_code import SubmitTimePartitionKey
from .user_code import UserCode
from .user_code import UserVerifyKeyPartitionKey


@instrument
@serializable()
class UserCodeStash(BaseUIDStoreStash):
    object_type = UserCode
    settings: PartitionSettings = PartitionSettings(
        name=UserCode.__canonical_name__, object_type=UserCode
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @catch(StashNotFoundException, StashException)
    def get_all_by_user_verify_key(
        self, credentials: SyftVerifyKey, user_verify_key: SyftVerifyKey
    ) -> list[UserCode]:
        qks = QueryKeys(qks=[UserVerifyKeyPartitionKey.with_obj(user_verify_key)])
        # when query one is adjusted, we'd do return self.query_one(...).unwrap()
        match self.query_one(credentials=credentials, qks=qks):
            case Ok(user_code_list):
                return cast(list[UserCode], [user_code_list])
            case Ok(None):
                raise StashNotFoundException(
                    f"UserCode not found for user {user_verify_key}"
                )
            case Err(msg):
                raise StashException(msg)
            case _:
                raise StashException("Unknown error")

    @catch(StashNotFoundException, StashException)
    def get_by_code_hash(self, credentials: SyftVerifyKey, code_hash: str) -> UserCode:
        qks = QueryKeys(qks=[CodeHashPartitionKey.with_obj(code_hash)])
        match self.query_one(credentials=credentials, qks=qks):
            case Ok(user_code):
                return cast(UserCode, user_code)
            case Ok(None):
                raise StashNotFoundException(
                    f"UserCode not found with hash {code_hash}"
                )
            case Err(msg):
                raise StashException(msg)
            case _:
                raise StashException("Unknown error")

    @catch(StashNotFoundException, StashException)
    def get_by_service_func_name(
        self, credentials: SyftVerifyKey, service_func_name: str
    ) -> list[UserCode]:
        qks = QueryKeys(qks=[ServiceFuncNamePartitionKey.with_obj(service_func_name)])
        match self.query_all(
            credentials=credentials, qks=qks, order_by=SubmitTimePartitionKey
        ):
            case Ok(user_code_list):
                return cast(list[UserCode], user_code_list)
            case Ok(None):
                raise StashNotFoundException(
                    f"UserCode not found for service function {service_func_name}"
                )
            case Err(msg):
                raise StashException(msg)
            case _:
                raise StashException("Unknown error")

    @catch(StashNotFoundException, StashException)
    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> UserCode:
        match self.get_by_uid(credentials, uid):
            case Ok(user_code):
                return cast(UserCode, user_code)
            case Ok(None):
                raise StashNotFoundException(f"UserCode not found for uid {uid}")
            case Err(msg):
                raise StashException(msg)
            case _:
                raise StashException("Unknown error")

    @catch(StashException)
    def set(self, credentials: SyftVerifyKey, user_code: UserCode) -> UserCode:
        match super().set(credentials=credentials, obj=user_code):
            case Ok(result):
                return cast(UserCode, result)
            case Err(msg):
                raise StashException(msg)
            case _:
                raise StashException("Unknown error")
