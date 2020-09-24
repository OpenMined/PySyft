# third party
from nacl.signing import SigningKey

# syft relative
from ...core.common.message import SyftMessage
from ...core.io.address import Address
from ...core.io.connection import ClientConnection
from ...core.io.route import SoloRoute
from ...core.node.common.client import Client
from ...decorators.syft_decorator_impl import syft_decorator
from ..services.signaling_service import RegisterNewPeerMessage


class SignalingClient(object):
    def __init__(
        self, url: str, conn_type: ClientConnection, client_type: Client
    ) -> None:
        # Load an Signing Key instance
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key

        # Use Signaling Server metadata
        # to build client route
        conn = conn_type(url=url)  # type: ignore
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
        signaling_client = client_type(  # type: ignore
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

    @syft_decorator(typechecking=True)
    def __register(self) -> None:
        _response = self.__client.send_immediate_msg_with_reply(
            msg=RegisterNewPeerMessage(
                address=self.__client.address, reply_to=self.__client.address
            )
        )
        self.duet_id = _response.peer_id

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(self, msg: SyftMessage) -> SyftMessage:
        return self.__client.send_immediate_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(self, msg: SyftMessage) -> None:
        self.__client.send_immediate_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(self, msg: SyftMessage) -> None:
        self.__client.send_eventual_msg_without_reply(msg=msg)
