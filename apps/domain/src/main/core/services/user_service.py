# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey


# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.user_messages import (
    CreateUserMessage,
    CreateUserResponse,
    GetUserMessage,
    GetUserResponse,
    UpdateUserMessage,
    UpdateUserResponse,
    DeleteUserMessage,
    DeleteUserResponse,
    GetUsersMessage,
    GetUsersResponse,
    SearchUsersMessage,
    SearchUsersResponse,
)

from ..exceptions import MissingRequestKeyError, RoleNotFoundError, AuthorizationError
from ..database.utils import model_to_json


@syft_decorator(typechecking=True)
def create_user_msg(
    msg: CreateUserMessage,
    node: AbstractNode,
) -> CreateUserResponse:

    # Get Payload Content
    _email = msg.content.get("email", None)
    _password = msg.content.get("password", None)
    _current_user_id = msg.content.get("current_user", None)
    _role = msg.content.get("role", None)

    users = node.users

    # Default response status
    _success = True
    _msg_field = "msg"
    _msg = ""
    _admin_role = node.roles.query(name="Owner")[0]

    # Check if email/password fields are empty
    if not _email or not _password:
        _success = False
        _msg_field = "error"
        _msg = "Invalid request payload, empty fields (email/password)!"
    # 1 - First User
    elif not len(users):
        # Use Domain Root Key
        _node_private_key = node.signing_key.encode(encoder=HexEncoder).decode("utf-8")
        users.signup(
            email=_email,
            password=_password,
            role=_admin_role.id,
            private_key=_node_private_key,
        )
    # 2 - Create User with custom role (Permission required)
    elif (
        _role
        and users.contain(id=_current_user_id)
        and users.role(user_id=_current_user_id).can_create_users
    ):
        _success = node.roles.contain(name=_role) and _role != _admin_role.name
        if _success:
            # Generate a new signing key
            _private_key = SigningKey.generate()
            users.signup(
                email=_email,
                password=_password,
                role=node.roles.query(name=_role)[0].id,
                private_key=_private_key.encode(encoder=HexEncoder).decode("utf-8"),
            )
        # If role name not found
        elif not node.roles.contain(name=_role):
            _msg = "Role not found!"
        # If purposed role is Owner
        elif _role == _admin_role.name:
            _msg = 'You can\'t create a new User with "Owner" role!'
    # 3 - Create default user (without custom role)
    else:
        # Generate a new signing key
        _private_key = SigningKey.generate()
        users.signup(
            email=_email,
            password=_password,
            role=node.roles.query(name="User")[0].id,
            private_key=_private_key.encode(encoder=HexEncoder).decode("utf-8"),
        )

    if _success:
        _msg = "User created successfully!"
    else:
        _msg_field = "error"

    return CreateUserResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


@syft_decorator(typechecking=True)
def update_user_msg(
    msg: UpdateUserMessage,
    node: AbstractNode,
) -> UpdateUserResponse:

    # Default response status
    _success = True
    _msg_field = "msg"
    _msg = ""

    # Get Payload Content
    _user_id = msg.content.get("user_id", None)
    _email = msg.content.get("email", None)
    _password = msg.content.get("password", None)
    _current_user_id = msg.content.get("current_user", None)
    _role = msg.content.get("role", None)
    _groups = msg.content.get("groups", None)

    users = node.users

    # If user ID not found
    if not users.contain(id=_user_id):
        _success = False
        _msg_field = "User ID not found!"

    # Change Email Request
    elif _email:
        users.set(user_id=_user_id, email=_email)
    # Change Password Request
    elif _password:
        users.set(user_id=_user_id, email=_email)
    # Change Role Request
    elif _role:
        _success = (
            node.roles.contain(name=_role)
            and _role != "Owner"
            and node.permissions.can_create_users(user_id=_current_user_id)
        )
        # If all premises were respected
        if _success:
            new_role_id = node.roles.query(name=_role)[0].id
            users.set(user_id=_user_id, role=new_role_id)
        # If they weren't respected
        elif not node.roles.contain(name=_role):
            _msg = "Role not found!"
        elif _role == "Owner":
            _msg = "You can't change it to Owner role!"
        else:
            _msg = "You're not allowed to change User roles!"
    # Change group
    elif _groups:
        _success = node.groups.contain(
            name=_groups
        ) and node.permissions.can_create_users(user_id=_current_user_id)
        # If all premises were respected
        if _success:
            new_group_id = node.groups.query(name=_role)[0].id
            node.groups.set(user_id=_user_id, group_id=new_group_id)
        # If they weren't respected
        elif not node.groups.contain(name=_groups):
            _msg = "Group not found!"
        elif not node.permissions.can_create_users(user_id=_current_user_id):
            _msg = "You're not allowed to change User groups!"

    if _success:
        _msg = "User updated successfully!"
    else:
        _msg_field = "error"

    return UpdateUserResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


@syft_decorator(typechecking=True)
def get_user_msg(
    msg: GetUserMessage,
    node: AbstractNode,
) -> GetUserResponse:

    # Default response status
    _success = True
    _msg_field = "user"
    _msg = ""

    # Get Payload Content
    _user_id = msg.content.get("user_id", None)
    _current_user_id = msg.content.get("current_user", None)

    _success = node.users.contain(id=_user_id) and node.permissions.can_triage_requests(
        user_id=_current_user_id
    )

    if _success:
        user = node.users.query(id=_user_id)[0]
    elif node.users.contain(id=_user_id):
        _msg = "User ID not found!"
    else:
        _msg = "You're not allowed to get User information!"

    if _success:
        _msg = model_to_json(user)
    else:
        _msg_field = "error"

    return GetUserResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


@syft_decorator(typechecking=True)
def get_all_users_msg(
    msg: GetUsersMessage,
    node: AbstractNode,
) -> GetUsersResponse:
    # Default response status
    _success = True
    _msg_field = "users"
    _msg = ""

    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)

    _success = node.permissions.can_triage_requests(user_id=_current_user_id)

    if _success:
        users = node.users.all()
    else:
        _msg = "You're not allowed to get User information!"

    if _success:
        _msg = {user.id: model_to_json(user) for user in users}
    else:
        _msg_field = "error"
    return GetUsersResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


@syft_decorator(typechecking=True)
def del_user_msg(
    msg: DeleteUserMessage,
    node: AbstractNode,
) -> DeleteUserResponse:
    # Default response status
    _success = True
    _msg_field = "user"
    _msg = ""

    # Get Payload Content
    _user_id = msg.content.get("user_id", None)
    _current_user_id = msg.content.get("current_user", None)

    _success = node.users.contain(id=_user_id) and node.permissions.can_create_users(
        user_id=_current_user_id
    )

    if _success:
        user = node.users.delete(id=_user_id)
    elif node.users.contain(id=_user_id):
        _msg = "User ID not found!"
    else:
        _msg = "You're not allowed to delete User information!"

    if _success:
        _msg = "User deleted successfully"
    else:
        _msg_field = "error"

    return DeleteUserResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


@syft_decorator(typechecking=True)
def search_users_msg(
    msg: SearchUsersMessage,
    node: AbstractNode,
) -> SearchUsersResponse:

    # Default response status
    _success = True
    _msg_field = "users"
    _msg = ""

    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    _email = msg.content.get("email", None)
    _role = msg.content.get("role", None)
    _group = msg.content.get("group", None)

    _success = node.permissions.can_triage_requests(user_id=_current_user_id)

    if _success:
        users = node.users.query(email=_email, role=_role, group=_group)
    else:
        _msg = "You're not allowed to get User information!"

    if _success:
        _msg = {user.id: model_to_json(user) for user in users}
    else:
        _msg_field = "error"

    return DeleteUserResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


class UserManagerService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateUserMessage: create_user_msg,
        UpdateUserMessage: update_user_msg,
        GetUserMessage: get_user_msg,
        GetUsersMessage: get_all_users_msg,
        DeleteUserMessage: del_user_msg,
        SearchUsersMessage: search_users_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateUserMessage,
            UpdateUserMessage,
            GetUserMessage,
            GetUsersMessage,
            DeleteUserMessage,
            SearchUsersMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateUserResponse,
        UpdateUserResponse,
        GetUserResponse,
        GetUsersResponse,
        DeleteUserResponse,
        SearchUsersResponse,
    ]:
        return UserManagerService.msg_handler_map[type(msg)](msg=msg, node=node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateUserMessage,
            UpdateUserMessage,
            GetUserMessage,
            GetUsersMessage,
            DeleteUserMessage,
            SearchUsersMessage,
        ]
