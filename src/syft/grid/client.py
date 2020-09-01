# stdlib
import binascii
import json
import pickle

# third party
import requests

# syft absolute
import syft as sy

# syft relative
from ..core.common.message import EventualSyftMessageWithoutReply
from ..core.common.message import ImmediateSyftMessageWithReply
from ..core.common.message import ImmediateSyftMessageWithoutReply
from ..core.common.message import SyftMessage
from ..core.io.connection import ClientConnection
from ..core.io.route import SoloRoute


class GridHttpClientConnection(ClientConnection):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> object:
        reply = self.send_msg(msg)
        binary = binascii.unhexlify(json.loads(reply.text)["data"])
        return pickle.loads(binary)  # nosec # TODO make less insecure

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        self.send_msg(msg)

    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        self.send_msg(msg)

    def send_msg(self, msg: SyftMessage) -> requests.Response:
        data = pickle.dumps(msg).hex()
        r = requests.post(url=self.base_url + "recv", json={"data": data})
        return r


def connect(domain_url: str = "http://localhost:5000/") -> sy.DomainClient:
    binary = binascii.unhexlify(requests.get(domain_url).text)
    client_metadata = pickle.loads(binary)  # nosec # TODO make less insecure

    conn = GridHttpClientConnection(base_url=domain_url)
    address = client_metadata["address"]
    name = client_metadata["name"]
    id = client_metadata["id"]
    route = SoloRoute(destination=id, connection=conn)
    client = sy.DomainClient(address=address, name=name, routes=[route])
    return client
