# stdlib
from enum import Enum
from typing import Callable
from typing import Optional

# third party
from pydantic import BaseModel
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .credentials import SyftCredentials
from .credentials import SyftSigningKey
from .credentials import SyftVerifyKey


class ServiceRoleCapability(Enum):
    CAN_MAKE_DATA_REQUESTS = 1
    CAN_TRIAGE_DATA_REQUESTS = 2
    CAN_MANAGE_PRIVACY_BUDGET = 4
    CAN_CREATE_USERS = 8
    CAN_MANAGE_USERS = 16
    CAN_EDIT_ROLES = 32
    CAN_MANAGE_INFRASTRUCTURE = 64
    CAN_UPLOAD_DATA = 128
    CAN_UPLOAD_LEGAL_DOCUMENT = 256
    CAN_EDIT_DOMAIN_SETTINGS = 512


class ServiceRole(BaseModel):
    GUEST = 1


class User(SyftObject):
    # version
    __canonical_name__ = "User"
    __version__ = 1

    # fields
    email: str
    name: str
    hashed_password: str
    salt: str
    signing_key: SyftSigningKey
    verify_key: SyftVerifyKey
    role: ServiceRole
    created_at: Optional[str]

    # serde / storage rules
    __attr_state__ = [
        "email",
        "name",
        "hashed_password",
        "salt",
        "signing_key",
        "verify_key",
        "role",
        "created_at",
    ]
    __attr_searchable__ = ["name", "email", "verify_key"]
    __attr_unique__ = ["email", "signing_key"]

    # @service_method()
    # def create() -> Result[UserView, str]:
    #     pass

    # @permission()
    def delete(self, credentials: SyftCredentials) -> Result[bool, str]:
        pass

    # @permission()
    def delete_user(self, uid: UID, credentials: SyftCredentials) -> Result[bool, str]:
        pass

    # def update_user(user: UpdateUserView) -> Result[UserView, str]:
    #     pass


def transform(input: dict, output: dict, args):
    pass


def drop(list_keys) -> Callable:
    def drop_keys(output: dict) -> dict:
        for key in list_keys:
            del output[key]
        return output

    return drop_keys


def hash_password(output: dict):
    if output["password"] == output["password_verify"]:
        output["password_hash"] = hash(output["password"])
    return output


# @serializable()


class SyftObjectUpdate:
    pass


class UpdateUserView(SyftObjectUpdate):
    __canonical_name__ = "User"
    __version__ = 1

    uid: Optional[UID] = None  # create or update
    email: str
    name: str
    password: str
    password_verify: str

    transforms_to = [hash_password, drop(["password", "password_verify"])]
    transforms_from = [drop(["email", "name", "role"])]

    def transform_to(self) -> dict:
        output = dict(self)
        for transform in self.transforms:
            output = transform(output)
        return output

    def transform_from(self) -> dict:
        output = dict(self)
        for transform in self.transforms:
            output = transform(output)
        return output


# import syft
# client = sy.login

# client.users.delete(uid)
# user = client.users.get(uid)
# user.delete()


class UserCollection:
    def __init__(self) -> None:
        self.data = {}
        self.primary_keys = {}

    def set(
        self, uid: UID, credentials: SyftCredentials, syft_object: SyftObject
    ) -> Result[bool, str]:
        pass
        # self.data


class ServiceStore:
    def __init__(self) -> None:
        self.collections = {}

    def set(
        self, uid: UID, credentials: SyftCredentials, syft_object: SyftObject
    ) -> Result[bool, str]:
        self.collections


# ActionService

#     @permission(guest_allowed)
#     def get_pointer()

#     @permission(guest_allowed)
#     def run_action()
#         -> has_pmerssion(uid, credentials)


# class CRUD:
#     def update_or_inset()


# class Service

#     @service(permission=Union[can_create_users)
#     def create_user(user: UserType) -> PublicView


# def safe_readcsv(path) -> pd.DataFrame:
#     return pd.read_csv(santize_path(path))


# def construct_api(credential: SyftCredential) -> SyftAPI:
#     if user.role
#     API = {
#         sevices: {
#             user: {
#                 create: function(args: kwargs) -> Result[Ok, Err]
#                 delete: function(args: kwargs)
#             },
#             action: {
#                 ast: {
#                     pandas: {
#                         read_csv: safe_readcsv
#                     }
#                     torch: {

#                     }
#                     numpy: {}
#                 }
#                 store: {
#                     get_pointer: function()
#                 }
#             }
#             codejail: {
#                 __andsfkjsadfjk__my_func: get_func_from_code_jail()
#             }
#         }
#     }

#     return API


# def get_func_from_code_jail(uid: UID):
#     allowed_users = codejail[uid].permissions

#     @permission(allowed_users)
#     def execute_custom_function:
#         return codejail[uid]
#     return execute_custom_function

# def __andsfkjsadfjk__my_func()


# class Pointer(ActionStoreView):
#     public_shape: shape
#     ...
