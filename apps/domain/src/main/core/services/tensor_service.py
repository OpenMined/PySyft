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

from syft.grid.messages.tensor_messages import (
    CreateTensorMessage,
    CreateTensorResponse,
    GetTensorMessage,
    GetTensorResponse,
    UpdateTensorMessage,
    UpdateTensorResponse,
    DeleteTensorMessage,
    DeleteTensorResponse,
    GetTensorsMessage,
    GetTensorsResponse,
)


@syft_decorator(typechecking=True)
def create_tensor_msg(
    msg: CreateTensorMessage,
) -> CreateTensorResponse:
    return CreateTensorResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "tensor created succesfully!"},
    )


@syft_decorator(typechecking=True)
def update_tensor_msg(
    msg: UpdateTensorMessage,
) -> UpdateTensorResponse:
    return UpdateTensorResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "tensor changed succesfully!"},
    )


@syft_decorator(typechecking=True)
def get_tensor_msg(
    msg: GetTensorMessage,
) -> GetTensorResponse:
    return GetTensorResponse(
        address=msg.reply_to,
        success=True,
        content={
            "tensor": {
                "id": "5484626",
                "tags": ["tensor-a"],
                "description": "tensor sample",
            }
        },
    )


@syft_decorator(typechecking=True)
def get_tensors_msg(
    msg: GetTensorsMessage,
) -> GetTensorsResponse:
    return GetTensorsResponse(
        address=msg.reply_to,
        success=True,
        content={
            "tensors": [
                {
                    "id": "35654sad6ada",
                    "tags": ["tensor-a"],
                    "description": "tensor sample",
                },
                {
                    "id": "adfarf3f1af5",
                    "tags": ["tensor-b"],
                    "description": "tensor sample",
                },
                {
                    "id": "fas4e6e1fas",
                    "tags": ["tensor-c"],
                    "description": "tensor sample",
                },
            ]
        },
    )


@syft_decorator(typechecking=True)
def del_tensor_msg(
    msg: DeleteTensorMessage,
) -> DeleteTensorResponse:
    return DeleteTensorResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "tensor deleted succesfully!"},
    )


class RegisterTensorService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateTensorMessage: create_tensor_msg,
        UpdateTensorMessage: update_tensor_msg,
        GetTensorMessage: get_tensor_msg,
        GetTensorsMessage: get_tensors_msg,
        DeleteTensorMessage: del_tensor_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateTensorMessage,
            UpdateTensorMessage,
            GetTensorMessage,
            GetTensorsMessage,
            DeleteTensorMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateTensorResponse,
        UpdateTensorResponse,
        GetTensorResponse,
        GetTensorsResponse,
        DeleteTensorResponse,
    ]:
        return RegisterTensorService.msg_handler_map[type(msg)](msg=msg)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateTensorMessage,
            UpdateTensorMessage,
            GetTensorMessage,
            GetTensorsMessage,
            DeleteTensorMessage,
        ]
