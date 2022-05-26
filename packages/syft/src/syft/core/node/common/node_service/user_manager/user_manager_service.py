# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey

# relative
from .....common.message import ImmediateSyftMessageWithReply
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import UserNotFoundError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..success_resp_message import SuccessResponseMessage
from .user_messages import CreateUserMessage
from .user_messages import DeleteUserMessage
from .user_messages import GetCandidatesMessage
from .user_messages import GetCandidatesResponse
from .user_messages import GetUserMessage
from .user_messages import GetUserResponse
from .user_messages import GetUsersMessage
from .user_messages import GetUsersResponse
from .user_messages import ProcessUserCandidateMessage
from .user_messages import SearchUsersMessage
from .user_messages import SearchUsersResponse
from .user_messages import UpdateUserMessage


def create_user_msg(
    msg: CreateUserMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check if node requires daa document
    if node.setup.first(domain_name=node.name).daa and not msg.daa_pdf:
        raise AuthorizationError(
            message="You can't apply a new User without a DAA document!"
        )

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

    app_id = node.users.create_user_application(
        name=msg.name,
        email=msg.email,
        password=msg.password,
        daa_pdf=msg.daa_pdf,
        institution=msg.institution,
        website=msg.website,
        budget=msg.budget,
    )

    user_role_id = -1
    try:
        user_role_id = node.users.role(verify_key=verify_key).id
    except Exception as e:
        print("verify_key not in db", e)

    if node.roles.can_create_users(role_id=user_role_id):
        node.users.process_user_application(
            candidate_id=app_id, status="accepted", verify_key=verify_key
        )

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="User created successfully!",
    )


def accept_or_deny_candidate(
    msg: ProcessUserCandidateMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    if True:  # node.users.can_create_users(verify_key=verify_key):
        node.users.process_user_application(
            candidate_id=msg.candidate_id, status=msg.status, verify_key=verify_key
        )
    else:
        raise AuthorizationError(
            message="You're not allowed to create a new User using this email!"
        )

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="User application processed successfully!",
    )


def update_user_msg(
    msg: UpdateUserMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    _valid_parameters = (
        msg.email
        or (msg.password and msg.new_password)
        or msg.role
        or msg.groups
        or msg.name
        or msg.budget
    )
    _allowed = msg.user_id == 0 or node.users.can_create_users(verify_key=verify_key)
    # Change own information
    if msg.user_id == 0:
        msg.user_id = int(node.users.get_user(verify_key).id)  # type: ignore

    _valid_user = node.users.contain(id=msg.user_id)

    if not _valid_parameters:
        raise MissingRequestKeyError(
            "Missing json fields ( email,password,role,groups, name )"
        )

    if not _allowed:
        raise AuthorizationError("You're not allowed to change other user data!")

    if not _valid_user:
        raise UserNotFoundError

    if msg.institution:
        node.users.set(user_id=str(msg.user_id), institution=msg.institution)

    if msg.website:
        node.users.set(user_id=str(msg.user_id), website=msg.website)

    if msg.budget:
        node.users.set(user_id=str(msg.user_id), budget=msg.budget)

    # Change Email Request
    if msg.email:
        node.users.set(user_id=str(msg.user_id), email=msg.email)

    # Change Password Request
    if msg.password and msg.new_password:
        node.users.change_password(
            user_id=str(msg.user_id),
            current_pwd=msg.password,
            new_pwd=msg.new_password,
        )

    # Change Name Request
    if msg.name:
        node.users.set(user_id=str(msg.user_id), name=msg.name)

    # Change Role Request
    if msg.role:
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
            node.users.set(user_id=msg.user_id, role=new_role_id)  # type: ignore
        elif target_user.role == node.roles.owner_role.id:
            raise AuthorizationError("You're not allowed to change Owner user roles!")
        else:
            raise AuthorizationError("You're not allowed to change User roles!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="User updated successfully!",
    )


def get_user_msg(
    msg: GetUserMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetUserResponse:
    # Check key permissions
    _allowed = node.users.can_triage_requests(verify_key=verify_key)
    if not _allowed:
        raise AuthorizationError(
            "get_user_msg You're not allowed to get User information!"
        )
    else:
        # Extract User Columns
        user = node.users.first(id=msg.user_id)
        _msg = model_to_json(user)

        # Use role name instead of role ID.
        _msg["role"] = node.roles.first(id=_msg["role"]).name

        # Remove private key
        del _msg["private_key"]

        # Get budget spent
        _msg["budget_spent"] = node.users.get_budget_for_user(
            verify_key=VerifyKey(user.verify_key.encode("utf-8"), encoder=HexEncoder)
        )

    return GetUserResponse(
        address=msg.reply_to,
        content=_msg,
    )


def get_all_users_msg(
    msg: GetUsersMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetUsersResponse:
    # Check key permissions
    _allowed = node.users.can_triage_requests(verify_key=verify_key)
    if not _allowed:
        raise AuthorizationError(
            "get_all_users_msg You're not allowed to get User information!"
        )
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

            # Remaining Budget
            # TODO:
            # Rename it from budget_spent to remaining budget
            _user_json["budget_spent"] = node.users.get_budget_for_user(  # type: ignore
                verify_key=VerifyKey(
                    user.verify_key.encode("utf-8"), encoder=HexEncoder
                ),
            )
            _msg.append(_user_json)

    return GetUsersResponse(
        address=msg.reply_to,
        content=_msg,
    )


def get_applicant_users(
    msg: GetCandidatesMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetCandidatesResponse:
    # Check key permissions
    _allowed = node.users.can_triage_requests(verify_key=verify_key)
    if not _allowed:
        raise AuthorizationError(
            "get_applicant_users You're not allowed to get User information!"
        )
    else:
        # Get All Users
        users = node.users.get_all_applicant()
        _msg = []
        _user_json = {}
        for user in users:
            _user_json = model_to_json(user)
            if user.daa_pdf:
                _user_json["daa_pdf"] = user.daa_pdf
            _msg.append(_user_json)

    return GetCandidatesResponse(
        address=msg.reply_to,
        content=_msg,
    )


def del_user_msg(
    msg: DeleteUserMessage,
    node: DomainInterface,
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
    node: DomainInterface,
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
            _msg = [model_to_json(user) for user in users]
        except UserNotFoundError:
            _msg = []
    else:
        raise AuthorizationError(
            "search_users_msg You're not allowed to get User information!"
        )

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
        Type[GetCandidatesMessage],
        Type[ProcessUserCandidateMessage],
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
        GetCandidatesResponse,
        SearchUsersResponse,
    ]

    msg_handler_map: Dict[INPUT_TYPE, Callable[..., OUTPUT_MESSAGES]] = {
        CreateUserMessage: create_user_msg,
        UpdateUserMessage: update_user_msg,
        GetUserMessage: get_user_msg,
        GetUsersMessage: get_all_users_msg,
        DeleteUserMessage: del_user_msg,
        SearchUsersMessage: search_users_msg,
        GetCandidatesMessage: get_applicant_users,
        ProcessUserCandidateMessage: accept_or_deny_candidate,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: DomainInterface,
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
            GetCandidatesMessage,
            ProcessUserCandidateMessage,
        ]
