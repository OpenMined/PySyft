# stdlib
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey

# relative
from ......proto.core.io.address_pb2 import Address as Address_PB
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.uid import UID
from ....domain.domain import Domain
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import ImmediateNodeServiceWithoutReply
from .object_transfer_messages import LoadObjectMessage
from .object_transfer_messages import LoadObjectResponse
from .object_transfer_messages import SaveObjectMessage
from .object_transfer_messages import SaveObjectResponse


class TransferObjectService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: Domain,
        msg: Union[
            LoadObjectMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[LoadObjectResponse, SaveObjectResponse]:
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
            searchable=True,
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
        node: Domain,
        msg: SaveObjectMessage,
        verify_key: VerifyKey,
    ) -> None:
        _obj_id = msg.content.get("uid", None)

        _syft_id = UID.from_string(value=_obj_id)

        try:
            _obj = node.store[_syft_id]  # noqa: 841
        except Exception:
            raise Exception("Object Not Found!")

        # TODO: this should call something using the 'node' variable. not executor directly (which is a pygrid op)
        # executor.submit(send_obj, _obj, node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithoutReply]]:
        return [SaveObjectMessage]
