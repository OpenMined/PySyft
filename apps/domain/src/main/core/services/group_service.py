# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.group_messages import (
    CreateGroupMessage,
    CreateGroupResponse,
    GetGroupMessage,
    GetGroupResponse,
    UpdateGroupMessage,
    UpdateGroupResponse,
    DeleteGroupMessage,
    DeleteGroupResponse,
    GetGroupsMessage,
    GetGroupsResponse,
)

from ..database.utils import model_to_json


def create_group_msg(
    msg: CreateGroupMessage,
    node: AbstractNode,
) -> CreateGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_name = msg.content.get("name", None)
    _users = msg.content.get("name", None)

    _success = True
    _msg_field = "msg"
    _msg = ""

    # Checks
    _is_allowed = node.users.role(user_id=_current_user_id).can_create_groups

    if not _group_name:
        _success = False
        _msg = "Invalid group name!"
    elif _is_allowed:
        node.groups.create(group_name=_group_name, users=_users)
    else:
        _success = False
        _msg = "You're not allowed to create groups!"

    if _success:
        _msg = "Group created successfully!"
    else:
        _msg_field = "error"

    return CreateGroupResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


def update_group_msg(
    msg: UpdateGroupMessage,
    node: AbstractNode,
) -> UpdateGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_id = msg.content.get("group_id", None)
    _group_name = msg.content.get("name", None)
    _users = msg.content.get("users", None)

    _success = True
    _msg_field = ""
    _msg = ""

    # Checks
    _is_allowed = node.users.role(user_id=_current_user_id).can_edit_groups

    if not node.groups.contain(id=_group_id):
        _success = False
        _msg = "Group ID not found!"
    elif _is_allowed:
        node.groups.update(group_id=_group_id, group_name=_group_name, users=_users)
    else:
        _success = False
        _msg = "You're not allowed to get this group!"

    if _success:
        _msg_field = "msg"
        _msg = "Group updated successfully!"
    else:
        _msg_field = "error"

    return UpdateGroupResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


def get_group_msg(
    msg: GetGroupMessage,
    node: AbstractNode,
) -> GetGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_id = msg.content.get("group_id", None)

    _success = True
    _msg_field = ""
    _msg = ""

    # Checks
    _is_allowed = node.users.role(user_id=_current_user_id).can_triage_requests

    if not node.groups.contain(id=_group_id):
        _success = False
        _msg = "Group ID not found!"
    elif _is_allowed:
        _group = node.groups.first(id=_group_id)
    else:
        _success = False
        _msg = "You're not allowed to get this group!"

    if _success:
        _msg = model_to_json(_group)
        _msg_field = "group"
    else:
        _msg_field = "error"

    return GetGroupResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
    )


def get_all_groups_msg(
    msg: GetGroupsMessage,
    node: AbstractNode,
) -> GetGroupsResponse:
    _current_user_id = msg.content.get("current_user", None)

    _success = True
    _msg_field = ""
    _msg = ""

    # Checks
    _is_allowed = node.users.role(user_id=_current_user_id).can_triage_requests

    if _is_allowed:
        _groups = node.groups.all()
    else:
        _success = False
        _msg = "You're not allowed to get the groups!"

    if _success:
        _msg = {group.id: model_to_json(group) for group in _groups}
        _msg_field = "groups"
    else:
        _msg_field = "error"

    return GetGroupsResponse(
        address=msg.reply_to,
        status_code=200,
        content={_msg_field: _msg},
    )


def del_group_msg(
    msg: DeleteGroupMessage,
    node: AbstractNode,
) -> DeleteGroupResponse:
    _current_user_id = msg.content.get("current_user", None)
    _group_id = msg.content.get("group_id", None)

    _success = True
    _msg_field = ""
    _msg = ""

    # Checks
    _is_allowed = node.users.role(user_id=_current_user_id).can_edit_groups

    if not node.groups.contain(id=_group_id):
        _success = False
        _msg = "Group ID not found!"
    elif _is_allowed:
        node.groups.delete(id=_group_id)
    else:
        _success = False
        _msg = "You're not allowed to delete this group!"

    if _success:
        _msg = "User deleted successfully!"
        _msg_field = "msg"
    else:
        _msg_field = "error"

    return DeleteGroupResponse(
        address=msg.reply_to,
        success=_success,
        content={_msg_field: _msg},
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
        return GroupManagerService.msg_handler_map[type(msg)](msg=msg, node=node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateGroupMessage,
            UpdateGroupMessage,
            GetGroupMessage,
            GetGroupsMessage,
            DeleteGroupMessage,
        ]
