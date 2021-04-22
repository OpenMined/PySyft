# stdlib
import json

# third party
import requests

# syft absolute
from syft import serialize
from syft.core.common.message import SignedEventualSyftMessageWithoutReply
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.message import SyftMessage
from syft.core.common.serde.deserialize import _deserialize
from syft.core.io.connection import ClientConnection
from syft.proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB

# syft relative
from ..client.enums import RequestAPIFields
from ..client.exceptions import RequestAPIException


class HTTPConnection(ClientConnection):
    def __init__(self, url: str) -> None:
        self.base_url = url

    def send_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        """
        Sends high priority messages and wait for their responses.

        This method implements a HTTP version of the
        ClientConnection.send_immediate_msg_with_reply

        :return: returns an instance of SignedImmediateSyftMessageWithReply.
        :rtype: SignedImmediateSyftMessageWithoutReply
        """

        # Serializes SignedImmediateSyftMessageWithReply
        # and send it using HTTP protocol
        response = self._send_msg(msg=msg)

        # Deserialize node's response
        if response.status_code == requests.codes.ok:
            response = _deserialize(blob=response.content, from_bytes=True)
        else:
            response_json = json.loads(response.content)
            raise RequestAPIException(response_json[RequestAPIFields.ERROR])

        # Return SignedImmediateSyftMessageWithoutReply
        return response

    def send_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        """
        Sends high priority messages without waiting for their reply.

        This method implements a HTTP version of the
        ClientConnection.send_immediate_msg_without_reply

        """
        # Serializes SignedImmediateSyftMessageWithoutReply
        # and send it using HTTP protocol
        self._send_msg(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        """
        Sends low priority messages without waiting for their reply.

        This method implements a HTTP version of the
        ClientConnection.send_eventual_msg_without_reply
        """
        # Serializes SignedEventualSyftMessageWithoutReply in json format
        # and send it using HTTP protocol
        self._send_msg(msg=msg)

    def _send_msg(self, msg: SyftMessage) -> requests.Response:
        """
        Serializes Syft messages in json format and send it using HTTP protocol.

        NOTE: Auxiliary method to avoid code duplication and modularity.

        :return: returns requests.Response object containing a JSON serialized
        SyftMessage
        :rtype: requests.Response
        """

        # Perform HTTP request using base_url as a root address
        r = requests.post(
            url=self.base_url,
            data=serialize(msg, to_bytes=True),
            headers={"Content-Type": "application/octet-stream"},
        )

        # Return request's response object
        # r.text provides the response body as a str
        return r

    def _get_metadata(self) -> Metadata_PB:
        """
        Request Node's metadata

        :return: returns node metadata
        :rtype: str of bytes
        """
        data: bytes = requests.get(self.base_url + "/metadata").content
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(data)
        return metadata_pb
