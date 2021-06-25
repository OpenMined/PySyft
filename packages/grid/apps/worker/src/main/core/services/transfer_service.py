# stdlib
import secrets
from threading import Thread
import traceback
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.uid import UID

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.node.domain.client import DomainClient
from syft.grid.client.client import connect
from syft.grid.client.grid_connection import GridHTTPConnection
from syft.grid.messages.transfer_messages import LoadObjectMessage
from syft.grid.messages.transfer_messages import LoadObjectResponse
from syft.grid.messages.transfer_messages import SaveObjectMessage
from syft.grid.messages.transfer_messages import SaveObjectResponse
from syft.proto.core.io.address_pb2 import Address as Address_PB
import torch as th

# grid relative
from ...utils.executor import executor


def send_obj(address, obj, node):
    client = connect(
        url=address, conn_type=GridHTTPConnection  # Domain Address
    )  # HTTP Connection Protocol
    y_s = obj.data.send(
        client, pointable=True, tags=obj.tags, description=obj.description
    )


class TransferObjectService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            LoadObjectMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[LoadObjectResponse, SaveObjectResponse,]:
        _worker_address = msg.content.get("address", None)
        _obj_id = msg.content.get("uid", None)
        _current_user_id = msg.content.get("current_user", None)

        users = node.users

        if not _current_user_id:
            _current_user_id = users.first(
                verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
            ).id

        addr_pb = Address_PB()
        addr_pb.ParseFromString(_worker_address.encode("ISO-8859-1"))
        _syft_address = _deserialize(blob=addr_pb)

        _syft_id = UID.from_string(value=_obj_id)

        _worker_client = node.in_memory_client_registry[_syft_address.domain_id]

        try:
            _obj = node.store[_syft_id]
        except Exception:
            raise Exception("Object Not Found!")

        _obj.data.send(
            _worker_client,
            pointable=True,
            tags=_obj.tags,
            description=_obj.description,
        )

        return LoadObjectResponse(
            address=msg.reply_to,
            status_code=200,
            content={"msg": "Object loaded successfully!"},
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [LoadObjectMessage]


class SaveObjectService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: SaveObjectMessage,
        verify_key: VerifyKey,
    ) -> None:
        _obj_id = msg.content.get("uid", None)
        _domain_addr = msg.content.get("domain_address", None)

        _syft_id = UID.from_string(value=_obj_id)

        try:
            _obj = node.store[_syft_id]
        except Exception:
            raise Exception("Object Not Found!")

        executor.submit(send_obj, _domain_addr, _obj, node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [SaveObjectMessage]
