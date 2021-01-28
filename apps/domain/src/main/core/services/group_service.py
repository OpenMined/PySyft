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
from syft.decorators.syft_decorator_impl import syft_decorator
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


@syft_decorator(typechecking=True)
def create_group_msg(
    msg: CreateGroupMessage,
) -> CreateGroupResponse:
    return CreateGroupResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Association request sent!"},
    )


@syft_decorator(typechecking=True)
def update_group_msg(
    msg: UpdateGroupMessage,
) -> UpdateGroupResponse:
    return UpdateGroupResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Association request received!"},
    )


@syft_decorator(typechecking=True)
def get_group_msg(
    msg: GetGroupMessage,
) -> GetGroupResponse:
    return GetGroupResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Association request was replied!"},
    )


@syft_decorator(typechecking=True)
def get_all_groups_msg(
    msg: GetGroupsMessage,
) -> GetGroupsResponse:
    return GetGroupsResponse(
        address=msg.reply_to,
        success=True,
        content={"association-request": {"ID": "51613546", "address": "156.89.33.200"}},
    )


@syft_decorator(typechecking=True)
def del_group_msg(
    msg: DeleteGroupMessage,
) -> DeleteGroupResponse:
    return DeleteGroupResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Association request deleted!"},
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
        return GroupManagerService.msg_handler_map[type(msg)](msg=msg)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateGroupMessage,
            UpdateGroupMessage,
            GetGroupMessage,
            GetGroupsMessage,
            DeleteGroupMessage,
        ]
