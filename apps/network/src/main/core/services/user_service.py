# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.grid.messages.user_messages import CreateUserMessage
from syft.grid.messages.user_messages import CreateUserResponse
from syft.grid.messages.user_messages import DeleteUserMessage
from syft.grid.messages.user_messages import DeleteUserResponse
from syft.grid.messages.user_messages import GetUserMessage
from syft.grid.messages.user_messages import GetUserResponse
from syft.grid.messages.user_messages import GetUsersMessage
from syft.grid.messages.user_messages import GetUsersResponse
from syft.grid.messages.user_messages import SearchUsersMessage
from syft.grid.messages.user_messages import SearchUsersResponse
from syft.grid.messages.user_messages import UpdateUserMessage
from syft.grid.messages.user_messages import UpdateUserResponse

# grid relative
from ..database import expand_user_object
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import MissingRequestKeyError
from ..exceptions import RoleNotFoundError
from ..exceptions import UserNotFoundError


def create_user_msg(
    msg: CreateUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> CreateUserResponse:

    # Get Payload Content
    _email = msg.content.get("email", None)
    _password = msg.content.get("password", None)
    _current_user_id = msg.content.get("current_user", None)
    _role = msg.content.get("role", None)

    users = node.users

    if not _current_user_id:
        try:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id
        except Exception:
            pass

    _admin_role = node.roles.first(name="Owner")

    # Check if email/password fields are empty
    if not _email or not _password:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (email/password)!"
        )

    # Check if this email was already registered
    try:
        users.first(email=_email)
        raise AuthorizationError(
            message="You can't create a new User using this email!"
        )
    except UserNotFoundError:
        pass

    # 1 - Owner Type
    # Create Owner type User (First user to be registered)
    # This user type will use node root key
    def create_owner_user():
        # Use Domain Root Key
        _node_private_key = node.signing_key.encode(encoder=HexEncoder).decode("utf-8")
        _verify_key = node.signing_key.verify_key.encode(encoder=HexEncoder).decode(
            "utf-8"
        )
        _user = users.signup(
            email=_email,
            password=_password,
            role=_admin_role.id,
            private_key=_node_private_key,
            verify_key=_verify_key,
        )
        return _user

    # 2 - Custom Type
    # Create a custom user (with a custom role)
    # This user can only be created by using an account with "can_create_users" permissions
    def create_custom_user():
        if _role != _admin_role.name:
            # Generate a new signing key
            _private_key = SigningKey.generate()
            _user = users.signup(
                email=_email,
                password=_password,
                role=node.roles.first(name=_role).id,
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
    def create_standard_user():
        # Generate a new signing key
        _private_key = SigningKey.generate()
        _user = users.signup(
            email=_email,
            password=_password,
            role=node.roles.first(name="User").id,
            private_key=_private_key.encode(encoder=HexEncoder).decode("utf-8"),
            verify_key=_private_key.verify_key.encode(encoder=HexEncoder).decode(
                "utf-8"
            ),
        )

    # Main logic
    if not len(users):
        create_owner_user()
    elif _role and users.can_create_users(user_id=_current_user_id):
        create_custom_user()
    else:
        create_standard_user()

    return CreateUserResponse(
        address=msg.reply_to,
        status_code=200,
        content={"message": "User created successfully!"},
    )


def update_user_msg(
    msg: UpdateUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> UpdateUserResponse:

    # Get Payload Content
    _user_id = msg.content.get("user_id", None)
    _email = msg.content.get("email", None)
    _password = msg.content.get("password", None)
    _current_user_id = msg.content.get("current_user", None)
    _role = msg.content.get("role", None)
    _groups = msg.content.get("groups", None)

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _valid_parameters = _email or _password or _role or _groups
    _allowed = int(_user_id) == int(_current_user_id) or users.can_create_users(
        user_id=_current_user_id
    )
    _valid_user = users.contain(id=_user_id)

    if not _valid_parameters:
        raise MissingRequestKeyError(
            "Missing json fields ( email,password,role,groups )"
        )

    if not _allowed:
        raise AuthorizationError("You're not allowed to change other user data!")

    if not _valid_user:
        raise UserNotFoundError

    # Change Email Request
    elif _email:
        users.set(user_id=_user_id, email=_email)

    # Change Password Request
    elif _password:
        users.set(user_id=_user_id, password=_password)

    # Change Role Request
    elif _role:
        _allowed = (
            node.roles.first(id=_role).name != "Owner"
            and users.can_create_users(user_id=_current_user_id)
            and users.role(user_id=_user_id)
            and users.role(user_id=_user_id).name != "Owner"
        )

        # If all premises were respected
        if _allowed:
            new_role_id = node.roles.first(id=_role).id
            users.set(user_id=_user_id, role=new_role_id)
        elif node.roles.first(id=_role).name == "Owner":
            raise AuthorizationError("You can't change it to Owner role!")
        elif users.role(user_id=_user_id).name == "Owner":
            raise AuthorizationError("You're not allowed to change Owner user roles!")
        else:
            raise AuthorizationError("You're not allowed to change User roles!")

    # Change group
    elif _groups:
        _allowed = users.can_create_users(user_id=_current_user_id)
        _valid_groups = (
            len(list(filter(lambda x: node.groups.first(id=x), _groups))) > 0
        )
        # If all premises were respected
        if _allowed and _valid_groups:
            node.groups.update_user_association(user_id=_user_id, groups=_groups)
        else:
            raise AuthorizationError("You're not allowed to change User groups!")

    return UpdateUserResponse(
        address=msg.reply_to,
        status_code=200,
        content={"message": "User updated successfully!"},
    )


def get_user_msg(
    msg: GetUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetUserResponse:
    # Get Payload Content
    _user_id = msg.content.get("user_id", None)
    _current_user_id = msg.content.get("current_user", None)

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    user = users.first(id=_user_id)
    _msg = model_to_json(user)

    return GetUserResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def get_all_users_msg(
    msg: GetUsersMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetUsersResponse:
    # Get Payload Content
    try:
        _current_user_id = msg.content.get("current_user", None)
    except Exception:
        _current_user_id = None

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    users = users.all()
    _msg = []
    for user in users:
        _user_json = model_to_json(user)
        _msg.append(_user_json)

    return GetUsersResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def del_user_msg(
    msg: DeleteUserMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteUserResponse:
    # Get Payload Content
    _user_id = msg.content.get("user_id", None)
    _current_user_id = msg.content.get("current_user", None)

    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    _allowed = (
        users.can_create_users(user_id=_current_user_id)
        and users.first(id=_user_id)
        and users.role(user_id=_user_id).name != "Owner"
    )
    if _allowed:
        node.users.delete(id=_user_id)
    else:
        raise AuthorizationError("You're not allowed to delete this user information!")

    return DeleteUserResponse(
        address=msg.reply_to,
        status_code=200,
        content={"message": "User deleted successfully!"},
    )


def search_users_msg(
    msg: SearchUsersMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SearchUsersResponse:
    # Get Payload Content
    _current_user_id = msg.content.get("current_user", None)
    users = node.users
    users = node.users
    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    user_parameters = {
        "email": msg.content.get("email", None),
        "role": msg.content.get("role", None),
    }

    filter_parameters = lambda key: user_parameters[key]
    filtered_parameters = filter(filter_parameters, user_parameters.keys())
    user_parameters = {key: user_parameters[key] for key in filtered_parameters}

    try:
        users = node.users.query(**user_parameters)
        _msg = [model_to_json(user) for user in users]
    except UserNotFoundError:
        _msg = {}

    return SearchUsersResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
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
        return UserManagerService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

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
