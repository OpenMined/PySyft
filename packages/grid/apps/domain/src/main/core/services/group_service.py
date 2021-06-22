# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.grid.messages.group_messages import CreateGroupMessage
from syft.grid.messages.group_messages import CreateGroupResponse
from syft.grid.messages.group_messages import DeleteGroupMessage
from syft.grid.messages.group_messages import DeleteGroupResponse
from syft.grid.messages.group_messages import GetGroupMessage
from syft.grid.messages.group_messages import GetGroupResponse
from syft.grid.messages.group_messages import GetGroupsMessage
from syft.grid.messages.group_messages import GetGroupsResponse
from syft.grid.messages.group_messages import UpdateGroupMessage
from syft.grid.messages.group_messages import UpdateGroupResponse

# grid relative
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import GroupNotFoundError
from ..exceptions import MissingRequestKeyError


def create_group_msg(
    msg: CreateGroupMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> CreateGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_name = msg.content.get("name", None)
    _users = msg.content.get("users", None)

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Checks
    _is_allowed = node.users.can_create_groups(user_id=_current_user_id)

    if not _group_name:
        raise MissingRequestKeyError("Invalid group name!")
    elif _is_allowed:
        node.groups.create(group_name=_group_name, users=_users)
    else:
        raise AuthorizationError("You're not allowed to create groups!")

    return CreateGroupResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Group created successfully!"},
    )


def update_group_msg(
    msg: UpdateGroupMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> UpdateGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_id = msg.content.get("group_id", None)
    _group_name = msg.content.get("name", None)
    _users = msg.content.get("users", None)

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Checks
    _is_allowed = node.users.can_create_groups(user_id=_current_user_id)

    if not node.groups.contain(id=_group_id):
        raise GroupNotFoundError("Group ID not found!")
    elif _is_allowed:
        node.groups.update(group_id=_group_id, group_name=_group_name, users=_users)
    else:
        raise AuthorizationError("You're not allowed to get this group!")

    return UpdateGroupResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Group updated successfully!"},
    )


def get_group_msg(
    msg: GetGroupMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_id = msg.content.get("group_id", None)

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Checks
    _is_allowed = node.users.can_create_groups(user_id=_current_user_id)

    if not node.groups.contain(id=_group_id):
        raise GroupNotFoundError("Group ID not found!")
    elif _is_allowed:
        _group = node.groups.first(id=_group_id)
    else:
        raise AuthorizationError("You're not allowed to get this group!")

    _msg = model_to_json(_group)
    _msg["users"] = node.groups.get_users(group_id=_group_id)

    return GetGroupResponse(
        address=msg.reply_to,
        status_code=200,
        content=_msg,
    )


def get_all_groups_msg(
    msg: GetGroupsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetGroupsResponse:

    try:
        _current_user_id = msg.content.get("current_user", None)
    except Exception:
        _current_user_id = None

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Checks
    _is_allowed = node.users.can_create_groups(user_id=_current_user_id)
    if _is_allowed:
        _groups = node.groups.all()
    else:
        raise AuthorizationError("You're not allowed to get the groups!")

    _groups = [model_to_json(group) for group in _groups]
    for group in _groups:
        group["users"] = node.groups.get_users(group_id=group["id"])

    return GetGroupsResponse(
        address=msg.reply_to,
        status_code=200,
        content=_groups,
    )


def del_group_msg(
    msg: DeleteGroupMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_id = msg.content.get("group_id", None)

    users = node.users

    if not _current_user_id:
        _current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Checks
    _is_allowed = node.users.can_create_groups(user_id=_current_user_id)

    if not node.groups.contain(id=_group_id):
        raise GroupNotFoundError("Group ID not found!")
    elif _is_allowed:
        node.groups.delete_association(group=_group_id)
        node.groups.delete(id=_group_id)
    else:
        raise AuthorizationError("You're not allowed to delete this group!")

    return DeleteGroupResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "User deleted successfully!"},
    )


class GroupManagerService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateGroupMessage: create_group_msg,
        UpdateGroupMessage: update_group_msg,
        GetGroupMessage: get_group_msg,
        GetGroupsMessage: get_all_groups_msg,
        DeleteGroupMessage: del_group_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateGroupMessage,
            UpdateGroupMessage,
            GetGroupMessage,
            GetGroupsMessage,
            DeleteGroupMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateGroupResponse,
        UpdateGroupResponse,
        GetGroupResponse,
        GetGroupsResponse,
        DeleteGroupResponse,
    ]:
        return GroupManagerService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateGroupMessage,
            UpdateGroupMessage,
            GetGroupMessage,
            GetGroupsMessage,
            DeleteGroupMessage,
        ]
