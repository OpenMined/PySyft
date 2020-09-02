import requests
import syft as sy

from syft.core.io.connection import ClientConnection

from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply, SyftMessage

from typing import Optional

class HTTPClient(ClientConnection):

    def __init__(self, base_url: str):
        self.base_url = base_url


    def send_immediate_msg_with_reply(
            self, msg: SignedImmediateSyftMessageWithReply, route: Optional[str] = "/"
    ) -> requests.Response:
        blob = self.send_msg(msg=msg,route=route).text
        if blob:
            response = sy.deserialize(blob=blob, from_json=True)
            return response

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply, route: Optional[str] = "/" ) -> None:
        self.send_msg(msg=msg,route=route)

    def send_eventual_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply, route: Optional[str] = "/" ) -> None:
        self.send_msg(msg=msg,route=route)

    def send_msg(self, msg: SyftMessage, route: str ) -> requests.Response:
        json_msg = msg.json()
        r = requests.post(url=self.base_url + route,  json=json_msg)
        return r
