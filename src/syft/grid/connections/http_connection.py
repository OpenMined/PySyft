# stdlib
import json
from typing import Dict
from typing import Tuple

# third party
import requests

# syft relative
from ...core.common.message import SignedEventualSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.common.serde.deserialize import _deserialize
from ...core.io.connection import ClientConnection
from ...decorators.syft_decorator_impl import syft_decorator


class HTTPConnection(ClientConnection):

    LOGIN_ROUTE = "/users/login"
    SYFT_ROUTE = "/pysyft"

    @syft_decorator(typechecking=True)
    def __init__(self, url: str) -> None:
        self.base_url = url

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """Sends high priority messages and wait for their responses.

        This method implements a HTTP version of the
        ClientConnection.send_immediate_msg_with_reply

        :return: returns an instance of SignedImmediateSyftMessageWithReply.
        :rtype: SignedImmediateSyftMessageWithoutReply
        """

        # Serializes SignedImmediateSyftMessageWithReply in json format
        # and send it using HTTP protocol
        blob = self._send_msg(msg=msg).text

        # Deserialize node's response
        response = _deserialize(blob=blob, from_json=True)

        # Return SignedImmediateSyftMessageWithoutReply
        return response

    @syft_decorator(typechecking=True)
    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """Sends high priority messages without waiting for their reply.

        This method implements a HTTP version of the
        ClientConnection.send_immediate_msg_without_reply

        """
        # Serializes SignedImmediateSyftMessageWithoutReply in json format
        # and send it using HTTP protocol
        self._send_msg(msg=msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """Sends low priority messages without waiting for their reply.

        This method implements a HTTP version of the
        ClientConnection.send_eventual_msg_without_reply
        """
        # Serializes SignedImmediateSyftMessageWithoutReply in json format
        # and send it using HTTP protocol
        self._send_msg(msg=msg)

    def _send_msg(self, msg: SyftMessage) -> requests.Response:
        """Serializes Syft messages in json format and send it using HTTP protocol.

        NOTE: Auxiliary method to avoid code duplication and modularity.

        :return: returns requests.Response object containing a JSON serialized
        SyftMessage
        :rtype: requests.Response
        """
        # Serialize SyftMessage object
        json_msg = msg.json()

        if self.session_token:
            header = {"token": self.session_token}
        else:
            header = {}

        # Perform HTTP request using base_url as a root address
        r = requests.post(
            url=self.base_url + HTTPConnection.SYFT_ROUTE, json=json_msg, headers=header
        )

        # Return request's response object
        # r.text provides the response body as a str
        return r

    def login(self, credentials: Dict) -> Tuple:
        response = requests.post(
            url=self.base_url + HTTPConnection.LOGIN_ROUTE, json=credentials
        )
        content = json.loads(response.text)
        if response.status_code != requests.codes.ok:
            raise Exception(content["error"])

        self.session_token = content["token"]
        return (content["metadata"], content["key"])

    @syft_decorator(typechecking=True)
    def _get_metadata(self) -> str:
        """Request Node's metadata

        :return: returns node metadata
        :rtype: str
        """
        return requests.get(self.base_url + "/metadata").text
