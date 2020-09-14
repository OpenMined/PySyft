# stdlib

# third party
import requests
from nacl.signing import SigningKey

# syft absolute
import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.message import SyftMessage
from syft.core.io.connection import ClientConnection
from syft.core.io.route import SoloRoute
from syft.core.node.network.client import NetworkClient


class HTTPClient(ClientConnection):
    def __init__(self):
        pass
    
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> requests.Response:
        blob = self.send_msg(msg=msg).text
        if blob:
            response = sy.deserialize(blob=blob, from_json=True)
            return response

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        self.send_msg(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        self.send_msg(msg=msg)

    def send_msg(self, msg: SyftMessage) -> requests.Response:
        json_msg = msg.json()
        r = requests.post(url=self.base_url, json=json_msg)
        return r

    def _get_metadata(self, url: str) -> str:
        self.base_url = url
        return requests.get(url + "/metadata").text

class GridNetworkClient(NetworkClient):

    def __init__(self,url: str, conn: ClientConnection):
        # generate a signing key
        self.signing_key = SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

        (
            spec_location,
            name,
            client_id,
        ) = NetworkClient.deserialize_client_metadata_from_node(metadata=conn._get_metadata(url))
        route = SoloRoute(destination=spec_location, connection=conn)
        print("\n\n\n\n name: ", name, "\n\n\n")
        super().__init__(
            network=spec_location,
            name=name,
            routes=[route],
            signing_key=self.signing_key,
            verify_key=self.verify_key,
        )
