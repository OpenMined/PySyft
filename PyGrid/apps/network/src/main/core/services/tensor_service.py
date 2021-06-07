# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft relative
from syft.core.common.group import VerifyAll
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.uid import UID
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.store.storeable_object import StorableObject
from syft.grid.messages.tensor_messages import CreateTensorMessage
from syft.grid.messages.tensor_messages import CreateTensorResponse
from syft.grid.messages.tensor_messages import DeleteTensorMessage
from syft.grid.messages.tensor_messages import DeleteTensorResponse
from syft.grid.messages.tensor_messages import GetTensorMessage
from syft.grid.messages.tensor_messages import GetTensorResponse
from syft.grid.messages.tensor_messages import GetTensorsMessage
from syft.grid.messages.tensor_messages import GetTensorsResponse
from syft.grid.messages.tensor_messages import UpdateTensorMessage
from syft.grid.messages.tensor_messages import UpdateTensorResponse
import torch as th


def create_tensor_msg(
    msg: CreateTensorMessage,
    node: AbstractNode,
) -> CreateTensorResponse:
    try:
        payload = msg.content

        new_tensor = th.tensor(payload["tensor"])
        new_tensor.tag(*payload.get("tags", []))
        new_tensor.describe(payload.get("description", ""))

        id_at_location = UID()

        # Step 2: create message which contains object to send
        storable = StorableObject(
            id=id_at_location,
            data=new_tensor,
            tags=new_tensor.tags,
            description=new_tensor.description,
            search_permissions={VerifyAll(): None}
            if payload.get("searchable", False)
            else {},
        )

        obj_msg = SaveObjectAction(obj=storable, address=node.address)

        signed_message = obj_msg.sign(
            signing_key=SigningKey(
                payload["internal_key"].encode("utf-8"), encoder=HexEncoder
            )
        )

        node.recv_immediate_msg_without_reply(msg=signed_message)

        return CreateTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={
                "msg": "Tensor created succesfully!",
                "tensor_id": str(id_at_location.value),
            },
        )
    except Exception as e:
        return CreateTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={"error": str(e)},
        )


def update_tensor_msg(
    msg: UpdateTensorMessage,
    node: AbstractNode,
) -> UpdateTensorResponse:
    try:
        payload = msg.content

        new_tensor = th.tensor(payload["tensor"])
        new_tensor.tag(*payload.get("tags", []))
        new_tensor.describe(payload.get("description", ""))

        key = UID.from_string(value=payload["tensor_id"])

        # Step 2: create message which contains object to send
        storable = StorableObject(
            id=key,
            data=new_tensor,
            tags=new_tensor.tags,
            description=new_tensor.description,
            search_permissions={VerifyAll(): None}
            if payload.get("searchable", False)
            else {},
        )

        obj_msg = SaveObjectAction(obj=storable, address=node.address)

        signed_message = obj_msg.sign(
            signing_key=SigningKey(
                payload["internal_key"].encode("utf-8"), encoder=HexEncoder
            )
        )

        node.recv_immediate_msg_without_reply(msg=signed_message)

        return UpdateTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={"msg": "Tensor modified succesfully!"},
        )
    except Exception as e:
        return UpdateTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={"error": str(e)},
        )


def get_tensor_msg(
    msg: GetTensorMessage,
    node: AbstractNode,
) -> GetTensorResponse:
    try:
        payload = msg.content

        # Retrieve the dataset from node.store
        key = UID.from_string(value=payload["tensor_id"])
        tensor = node.store[key]
        return GetTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={
                "tensor": {
                    "id": payload["tensor_id"],
                    "tags": tensor.tags,
                    "description": tensor.description,
                }
            },
        )
    except Exception as e:
        return GetTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={"error": str(e)},
        )


def get_tensors_msg(
    msg: GetTensorsMessage,
    node: AbstractNode,
) -> GetTensorsResponse:
    try:
        tensors = node.store.get_objects_of_type(obj_type=th.Tensor)

        result = []

        for tensor in tensors:
            result.append(
                {
                    "id": str(tensor.id.value),
                    "tags": tensor.tags,
                    "description": tensor.description,
                }
            )
        return GetTensorsResponse(
            address=msg.reply_to,
            status_code=200,
            content={"tensors": result},
        )
    except Exception as e:
        return GetTensorsResponse(
            address=msg.reply_to, success=False, content={"error": str(e)}
        )


def del_tensor_msg(
    msg: DeleteTensorMessage,
    node: AbstractNode,
) -> DeleteTensorResponse:
    try:
        payload = msg.content

        # Retrieve the dataset from node.store
        key = UID.from_string(value=payload["tensor_id"])
        node.store.delete(key=key)

        return DeleteTensorResponse(
            address=msg.reply_to,
            status_code=200,
            content={"msg": "Tensor deleted successfully!"},
        )
    except Exception as e:
        return DeleteTensorResponse(
            address=msg.reply_to,
            success=False,
            content={"error": str(e)},
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
        return RegisterTensorService.msg_handler_map[type(msg)](msg=msg, node=node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateTensorMessage,
            UpdateTensorMessage,
            GetTensorMessage,
            GetTensorsMessage,
            DeleteTensorMessage,
        ]
