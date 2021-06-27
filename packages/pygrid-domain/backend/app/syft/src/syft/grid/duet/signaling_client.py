# stdlib
from typing import Any
from typing import Type
from typing import Union

# third party
from nacl.signing import SigningKey

# syft relative
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.io.address import Address
from ...core.io.route import SoloRoute
from ...core.node.network.client import NetworkClient
from ..services.signaling_service import RegisterNewPeerMessage


class SignalingClient(object):
    def __init__(
        self, url: str, conn_type: type, client_type: Type[NetworkClient]
    ) -> None:
        # Load an Signing Key instance
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key

        # Use Signaling Server metadata
        # to build client route
        conn = conn_type(url=url)
        (
            spec_location,
            name,
            client_id,
        ) = client_type.deserialize_client_metadata_from_node(
            metadata=conn._get_metadata()
        )

        # Create a new Solo Route using the selected connection type
        route = SoloRoute(destination=spec_location, connection=conn)

        # Create a new signaling client using the selected client type
        signaling_client = client_type(
            network=spec_location,
            name=name,
            routes=[route],
            signing_key=signing_key,
            verify_key=verify_key,
        )

        self.__client = signaling_client
        self.__register()

    @property
    def address(self) -> Address:
        return self.__client.address

    def __register(self) -> None:
        _response = self.__client.send_immediate_msg_with_reply(
            msg=RegisterNewPeerMessage(
                address=self.__client.address, reply_to=self.__client.address
            )
        )
        self.duet_id = _response.peer_id  # type: ignore

    def send_immediate_msg_with_reply(
        self,
        msg: Union[SignedImmediateSyftMessageWithReply, ImmediateSyftMessageWithReply],
    ) -> SyftMessage:
        return self.__client.send_immediate_msg_with_reply(msg=msg)

    def send_immediate_msg_without_reply(
        self,
        msg: Union[
            SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply
        ],
    ) -> None:
        self.__client.send_immediate_msg_without_reply(msg=msg)

    def send_eventual_msg_without_reply(self, msg: Any) -> None:
        self.__client.send_eventual_msg_without_reply(msg=msg)
