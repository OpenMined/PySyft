# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node_service.auth import service_auth
from syft.core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)
from syft.lib.python import Dict as SyftDict
from syft.lib.python import List as SyftList

# relative
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import UserNotFoundError
from ...node_table.utils import model_to_json
from ..success_resp_message import SuccessResponseMessage
from ..user_manager.user_messages import CreateUserMessage
from ..user_manager.user_messages import DeleteUserMessage
from ..user_manager.user_messages import GetUserMessage
from ..user_manager.user_messages import GetUserResponse
from ..user_manager.user_messages import GetUsersMessage
from ..user_manager.user_messages import GetUsersResponse
from ..user_manager.user_messages import SearchUsersMessage
from ..user_manager.user_messages import SearchUsersResponse
from ..user_manager.user_messages import UpdateUserMessage


def create_user_msg(
    msg: CreateUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:

    # Check if email/password fields are empty
    if not msg.email or not msg.password:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (email/password)!"
        )

    # Check if this email was already registered
    try:
        node.users.first(email=msg.email)
        # If the email has already been registered, raise exception
        raise AuthorizationError(
            message="You can't create a new User using this email!"
        )
    except UserNotFoundError:
        # If email not registered, a new user can be created.
        pass

    # 2 - Custom Type
    # Create a custom user (with a custom role)
    # This user can only be created by using an account with "can_create_users" permissions
    def create_custom_user() -> None:
        _owner_role = node.roles.owner_role
        if msg.role != _owner_role.name:
            # Generate a new signing key
            _private_key = SigningKey.generate()
            node.users.signup(
                name=msg.name,
                email=msg.email,
                password=msg.password,
                budget=msg.budget,
                role=node.roles.first(name=msg.role).id,
                private_key=_private_key.encode(encoder=HexEncoder).decode("utf-8"),
                verify_key=_private_key.verify_key.encode(encoder=HexEncoder).decode(
                    "utf-8"
                ),
            )
        # If purposed role is Owner
        else:
            raise AuthorizationError(
                message='You can\'t create a new User with "Owner" role!'
            )

    # 3 - Standard type
    # Create a common user with no special permissions
    def create_standard_user() -> None:

        # Generate a new signing key
        _private_key = SigningKey.generate()

        encoded_pk = _private_key.encode(encoder=HexEncoder).decode("utf-8")
        encoded_vk = _private_key.verify_key.encode(encoder=HexEncoder).decode("utf-8")

        node.users.signup(
            name=msg.name,
            email=msg.email,
            password=msg.password,
            budget=msg.budget,
            role=node.roles.first(name="Data Scientist").id,
            private_key=encoded_pk,
            verify_key=encoded_vk,
        )

    # Main logic
    _allowed = node.users.can_create_users(verify_key=verify_key)

    if msg.role and _allowed:
        create_custom_user()
    else:
        create_standard_user()

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="User created successfully!",
    )


def update_user_msg(
    msg: UpdateUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    _valid_parameters = (
        msg.email or msg.password or msg.role or msg.groups or msg.name or msg.budget
    )
    _same_user = int(node.users.get_user(verify_key).id) == msg.user_id
    _allowed = _same_user or node.users.can_create_users(verify_key=verify_key)

    _valid_user = node.users.contain(id=msg.user_id)

    if not _valid_parameters:
        raise MissingRequestKeyError(
            "Missing json fields ( email,password,role,groups, name )"
        )

    if not _allowed:
        raise AuthorizationError("You're not allowed to change other user data!")

    if not _valid_user:
        raise UserNotFoundError

    # Change Email Request
    elif msg.email:
        node.users.set(user_id=msg.user_id, email=msg.email)

    # Change Password Request
    elif msg.password:
        node.users.set(user_id=msg.user_id, password=msg.password)

    # Change Name Request
    elif msg.name:
        node.users.set(user_id=msg.user_id, name=msg.name)

    # Change budget Request
    elif msg.budget:
        node.users.set(user_id=msg.user_id, budget=msg.budget)

    # Change Role Request
    elif msg.role:
        target_user = node.users.first(id=msg.user_id)
        _allowed = (
            msg.role != node.roles.owner_role.name  # Target Role != Owner
            and target_user.role
            != node.roles.owner_role.id  # Target User Role != Owner
            and node.users.can_create_users(verify_key=verify_key)  # Key Permissions
        )

        # If all premises were respected
        if _allowed:
            new_role_id = node.roles.first(name=msg.role).id
            node.users.set(user_id=msg.user_id, role=new_role_id)
        elif msg.role == node.roles.owner_role.name:
            raise AuthorizationError("You can't change it to Owner role!")
        elif target_user.role == node.roles.owner_role.id:
            raise AuthorizationError("You're not allowed to change Owner user roles!")
        else:
            raise AuthorizationError("You're not allowed to change User roles!")

    # Change group
    elif msg.groups:
        _allowed = node.users.can_create_users(verify_key=verify_key)
        _valid_groups = (
            len(list(filter(lambda x: node.groups.first(id=x), msg.groups))) > 0
        )
        # If all premises were respected
        if _allowed and _valid_groups:
            node.groups.update_user_association(user_id=msg.user_id, groups=msg.groups)
        else:
            raise AuthorizationError("You're not allowed to change User groups!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="User updated successfully!",
    )


def get_user_msg(
    msg: GetUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetUserResponse:
    # Check key permissions
    _allowed = node.users.can_triage_requests(verify_key=verify_key)
    if not _allowed:
        raise AuthorizationError("You're not allowed to get User information!")
    else:
        # Extract User Columns
        user = node.users.first(id=msg.user_id)
        _msg = model_to_json(user)

        # Use role name instead of role ID.
        _msg["role"] = node.roles.first(id=_msg["role"]).name

        # Remove private key
        del _msg["private_key"]

        # Add User groups
        _msg["groups"] = [
            node.groups.first(id=group).name
            for group in node.groups.get_groups(user_id=msg.user_id)
        ]

        # Get budget spent
        _msg["budget_spent"] = node.acc.user_budget(
            user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder)
        )

    return GetUserResponse(
        address=msg.reply_to,
        content=SyftDict(_msg),
    )


def get_all_users_msg(
    msg: GetUsersMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetUsersResponse:
    # Check key permissions
    _allowed = node.users.can_triage_requests(verify_key=verify_key)
    if not _allowed:
        raise AuthorizationError("You're not allowed to get User information!")
    else:
        # Get All Users
        users = node.users.all()
        _msg = []
        for user in users:
            _user_json = model_to_json(user)

            # Use role name instead of role ID.
            _user_json["role"] = node.roles.first(id=_user_json["role"]).name

            # Remove private key
            del _user_json["private_key"]

            # Add User groups
            _user_json["groups"] = [
                node.groups.first(id=group).name
                for group in node.groups.get_groups(user_id=user.id)
            ]

            _user_json["budget_spent"] = node.acc.user_budget(
                user_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder)
            )
            _msg.append(_user_json)

    return GetUsersResponse(
        address=msg.reply_to,
        content=SyftList(_msg),
    )


def del_user_msg(
    msg: DeleteUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:

    _target_user = node.users.first(id=msg.user_id)
    _not_owner = (
        node.roles.first(id=_target_user.role).name != node.roles.owner_role.name
    )

    _allowed = (
        node.users.can_create_users(verify_key=verify_key)  # Key Permission
        and _not_owner  # Target user isn't the node owner
    )
    if _allowed:
        node.users.delete(id=msg.user_id)
    else:
        raise AuthorizationError("You're not allowed to delete this user information!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="User deleted successfully!",
    )


def search_users_msg(
    msg: SearchUsersMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SearchUsersResponse:
    user_parameters = {
        "email": msg.email,
        "role": node.roles.first(name=msg.role).id,
    }

    filtered_parameters = filter(
        lambda key: user_parameters[key], user_parameters.keys()
    )
    user_parameters = {key: user_parameters[key] for key in filtered_parameters}

    _allowed = node.users.can_triage_requests(verify_key=verify_key)

    if _allowed:
        try:
            users = node.users.query(**user_parameters)
            if msg.groups:
                filtered_users = filter(
                    lambda x: node.groups.contain_association(
                        user=x.id, group=msg.group
                    ),
                    users,
                )
                _msg = [model_to_json(user) for user in filtered_users]
            else:
                _msg = [model_to_json(user) for user in users]
        except UserNotFoundError:
            _msg = []
    else:
        raise AuthorizationError("You're not allowed to get User information!")

    return SearchUsersResponse(
        address=msg.reply_to,
        content=_msg,
    )


class UserManagerService(ImmediateNodeServiceWithReply):
    INPUT_TYPE = Union[
        Type[CreateUserMessage],
        Type[UpdateUserMessage],
        Type[GetUserMessage],
        Type[GetUsersMessage],
        Type[DeleteUserMessage],
        Type[SearchUsersMessage],
    ]

    INPUT_MESSAGES = Union[
        CreateUserMessage,
        UpdateUserMessage,
        GetUserMessage,
        GetUsersMessage,
        DeleteUserMessage,
        SearchUsersMessage,
    ]

    OUTPUT_MESSAGES = Union[
        SuccessResponseMessage,
        GetUserResponse,
        GetUsersResponse,
        SearchUsersResponse,
    ]

    msg_handler_map: Dict[INPUT_TYPE, Callable[..., OUTPUT_MESSAGES]] = {
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
        msg: INPUT_MESSAGES,
        verify_key: VerifyKey,
    ) -> OUTPUT_MESSAGES:

        reply = UserManagerService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

        return reply

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
