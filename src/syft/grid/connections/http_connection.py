# stdlib

# third party
import requests

# syft absolute
import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.message import SyftMessage
from syft.core.common.serde.serializable import Serializable
from syft.core.io.connection import ClientConnection


class HTTPConnection(ClientConnection):
    def __init__(self, url: str) -> None:
        self.base_url = url

    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> Serializable:
        blob = self.send_msg(msg=msg).text

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

    def _get_metadata(self) -> str:
        return requests.get(self.base_url + "/metadata").text
