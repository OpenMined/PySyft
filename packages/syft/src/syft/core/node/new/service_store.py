# future
from __future__ import annotations

# third party
from result import Result

# relative
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.uid import UID
from .credentials import SyftCredentials


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


# class CRUD:
#     def update_or_inset()


# class Service:
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

# import syft as sy


# -> functional
# -> method

# client.service.action.create()

# client.service.user.create()

# user = client.service.user.get_uiser()

# user.delete()


# class User:

#     @service(permission=[GUEST, can_do_Stuff])
#     def delete(self: Union[UID, User, UserView]) -> Result[bool, err]:


# python client

# type(client.service.user) == class UserAPI


# class PublicAPI:
#     @staticmethod
#     def make_public_api_class(cls, class_)
#         inspect.signature(foo)
#         method = Dict[str, dict[]] = {
#             "create": {
#                 "args": [int, float]
#                 "kwargs": ["user_id": UID]
#             }
#         }

# >>> import inspect
# >>> print(inspect.signature(foo))
# >>> def foo(a: int, b: float, x='blah'):
# ...     pass
# >>> print(inspect.signature(foo))


# # @service_method()
# # def create() -> Result[UserView, str]:
# #     pass

# # @permission()
# def delete(self, credentials: SyftCredentials) -> Result[bool, str]:
#     pass

# # @permission()
# def delete_user(self, uid: UID, credentials: SyftCredentials) -> Result[bool, str]:
#     pass

# # def update_user(user: UpdateUserView) -> Result[UserView, str]:
# #     pass
